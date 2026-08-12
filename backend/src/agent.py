import logging
import os
import sqlite3  # INBUILT PYTHON DATABASE
from typing import Annotated

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    function_tool,  # IMPORTANT FOR DAY 4 TOOLS
    room_io,
    tokenize,
)
from livekit.plugins import deepgram, murf, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bol_khata.db")

# DAY 4 STRICT SYSTEM PROMPT (Memory, Consent & Devanagari Rules)
SYSTEM_PROMPT = """
IDENTITY: You are Anisha, a voice assistant for 'Bol-Khata' (Local Commerce track).
OBJECTIVES: Help vendors manage their orders and customers. You must follow the exact sequence below.

SEQUENCE (STRICT):
Step 1: When a user says Hi or greets you, ask for their name.
Step 2: Use the 'lookup_customer' tool to check their past record.
Step 3 (Returning Caller): If data is found, greet them by name and mention their past facts (e.g. past orders, usual quantities, preferred delivery slot).
Step 4 (New Caller): If not found, ask them for their past orders, usual quantities, and preferred delivery slot.
Step 5 (MANDATORY CONSENT): After gathering the facts, you MUST explicitly ask: "Can I save this information for next time?".
Step 6: ONLY if the user explicitly agrees, call the 'save_customer' tool. If they refuse, drop it and do not save.

DAY 5: INVENTORY & EXTERNAL DATA:
- When the user asks to check the stock or price of an item, you MUST call the `check_inventory` tool.
- GRACEFUL FAILURE: If the tool returns an ERROR (like timeout), DO NOT read the error code. Instead, gracefully apologize in natural language (e.g., "I am sorry, my stock system is currently down, I cannot check that right now.").
- TIMESTAMP MANDATE: When you successfully return stock or price data, you MUST tell the user when the data is from (e.g., "As of today's live rates, we have...").

DAY 7: HUMAN ESCALATION:
- TRIGGERS: If the user asks for a refund, complains about rotten/bad quality items, or has a payment dispute, you MUST stop trying to solve it yourself.
- MANDATORY CONSENT: Before calling `create_escalation`, you MUST explicitly ask: "Can I forward this issue to our human support team?"
- If they say yes, call the tool.
- NEXT STEPS: Once the tool returns the Ticket ID, you MUST tell the user their reference ID and assure them: "A human agent will contact you within 24 hours."

STRICT LANGUAGE & SCRIPT RULES:
1. STRICT 1-to-1 language matching:
   - If the user speaks English, reply ONLY in pure English. No Hindi or Hinglish.
   - If the user speaks Hindi, reply ONLY in pure Hindi using Devanagari script (e.g., नमस्ते). No English, Hinglish, or romanized Hindi.
   - Do NOT mix languages in a single response. Do NOT translate your response or add translations.
2. Keep responses short and conversational.

STRICT TOOL CALLING RULES:
1. Call tools silently. Do NOT announce that you are calling a tool, using a database, or invoking a function.
2. NEVER mention function or tool names like 'lookup_customer', 'save_customer', 'database', 'tool', or 'function' to the user.
3. Simply execute the tool behind the scenes, and then respond naturally once you receive the tool's result.
4. When calling `lookup_customer` or `save_customer`, you MUST pass ONLY the user's first name, in lowercase, with no punctuation (e.g., 'mehran').
5. DO NOT say the data is saved until the `save_customer` tool returns a success message.

CRITICAL RULE FOR SAVING DATA:
If the user says 'Yes' to saving their information, you MUST IMMEDIATELY trigger the `save_customer` tool.
DO NOT simply say 'I have saved it' verbally without triggering the tool.
You are strictly forbidden from confirming that the data is saved UNTIL you have actually executed the `save_customer` function and received a success response from it.
"""


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)
        self._init_db()

    # STEP 1: CREATE SQLITE DATABASE
    def _init_db(self):
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS customers
                         (name TEXT PRIMARY KEY, past_orders TEXT, usual_quantities TEXT, preferred_delivery_slot TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS tickets
                         (ticket_id TEXT PRIMARY KEY, customer_name TEXT, issue_summary TEXT, urgency TEXT)""")
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
        finally:
            if conn:
                conn.close()

    # STEP 2 & 3: LOOKUP TOOL (Agent calls this to find old callers)
    @function_tool
    async def lookup_customer(
        self, name: Annotated[str, "The customer's first name, in lowercase"]
    ):
        """Use this tool FIRST to search for an existing customer in the database by their name."""
        clean_name = name.strip().lower()
        logger.info(f"Looking up customer: {clean_name}")
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "SELECT past_orders, usual_quantities, preferred_delivery_slot FROM customers WHERE name=?",
                (clean_name,),
            )
            row = c.fetchone()
            if row:
                return f"Customer found: Past orders: {row[0]}. Usual quantities: {row[1]}. Preferred delivery slot: {row[2]}."
            else:
                return "Customer not found in database. Treat as a new customer."
        except sqlite3.Error as e:
            logger.error(f"Database lookup error for {clean_name}: {e}")
            return f"Error looking up customer {clean_name}."
        finally:
            if conn:
                conn.close()

    # STEP 2, 3 & 5: SAVE TOOL (Agent calls this to remember new data)
    @function_tool
    async def save_customer(
        self,
        name: Annotated[str, "The customer's first name, in lowercase"],
        past_orders: Annotated[str, "The items the customer usually orders"],
        usual_quantities: Annotated[str, "The quantities the customer usually orders"],
        preferred_delivery_slot: Annotated[
            str, "The customer's preferred delivery time"
        ],
    ):
        """Use this tool to save a new customer's record. YOU MUST ASK FOR CONSENT BEFORE USING THIS."""
        clean_name = name.strip().lower()
        logger.info(f"Saving customer: {clean_name}")
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?)",
                (
                    clean_name,
                    past_orders,
                    usual_quantities,
                    preferred_delivery_slot,
                ),
            )
            conn.commit()
            return f"Successfully saved record for {clean_name}."
        except sqlite3.Error as e:
            logger.error(f"Database save error for {clean_name}: {e}")
            return f"Error saving customer {clean_name}."
        finally:
            if conn:
                conn.close()

    # DAY 5: NEW INVENTORY TOOL
    @function_tool
    async def check_inventory(
        self, item_name: Annotated[str, "The name of the item to check stock for"]
    ):
        """Use this tool to check the stock and price of an item."""
        logger.info(f"Checking inventory for: {item_name}")

        inventory = {
            "potatoes": {"price": 30, "stock": 50},
            "onions": {"price": 40, "stock": 30},
            "tomatoes": {"price": 50, "stock": 20},
        }

        normalized_item = item_name.strip().lower()

        if normalized_item == "dragonfruit":
            return "ERROR: Database connection timeout. 503 Service Unavailable."

        if normalized_item in inventory:
            data = inventory[normalized_item]
            return f"Found {normalized_item}. Price: {data['price']} rupees/kg, Stock: {data['stock']} kg."
        else:
            return "Item not found in catalog."

    # DAY 7: NEW HUMAN ESCALATION TOOL
    @function_tool
    async def create_escalation(
        self,
        customer_name: Annotated[str, "The customer's first name"],
        issue_summary: Annotated[
            str,
            "A brief summary of the issue (e.g., refund, payment dispute, quality complaint)",
        ],
        urgency: Annotated[str, "The urgency of the issue (e.g., high, low)"],
    ):
        """Use this tool to escalate a complex issue (refund, payment dispute, quality complaint) to a human agent."""
        import random

        ticket_id = f"BK-{random.randint(1000, 9999)}"
        clean_name = customer_name.strip().lower()

        logger.info("\n======================================")
        logger.info(f"🚨 NEW TICKET ESCALATION: {ticket_id}")
        logger.info(f"Customer: {clean_name}")
        logger.info(f"Issue: {issue_summary}")
        logger.info(f"Urgency: {urgency}")
        logger.info("======================================\n")

        conn = None
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "INSERT INTO tickets (ticket_id, customer_name, issue_summary, urgency) VALUES (?, ?, ?, ?)",
                (ticket_id, clean_name, issue_summary, urgency),
            )
            conn.commit()
            return f"Successfully created ticket {ticket_id}."
        except sqlite3.Error as e:
            logger.error(f"Database error while creating ticket {ticket_id}: {e}")
            return "Error creating ticket."
        finally:
            if conn:
                conn.close()


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
        ),
        tts=murf.TTS(
            voice="Anisha",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
