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
    RunContext,
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

# MAIN AGENT (ANISHA) SYSTEM PROMPT
SYSTEM_PROMPT = """
IDENTITY: You are Anisha, a voice assistant for 'Bol-Khata' (Local Commerce track).

STRICT RULES & TOOL USAGE:
1. MEMORY (DAY 4): When a user greets you (e.g., "Hi, I am Mehran"), you MUST immediately silently call the `lookup_customer` tool.
- If found: Greet them with their past order details.
- If not found: Ask what they usually order, then ask "Can I save this?". If they say yes, you MUST call `save_customer`.
2. INVENTORY (DAY 5): If the user asks for the price, stock, or availability of an item (like onions or tomatoes), you MUST silently call the `check_inventory` tool. NEVER say you don't have access to inventory. Tell them the price based on the tool's result.
3. CALL ANALYTICS (DAY 8): If you successfully answer an inventory question, silently call `mark_call_successful`.
4. HANDOFF (DAY 9): If the user asks for a refund, return, or complains about bad/rotten quality (e.g., rotten potatoes), you MUST explicitly say "I will connect you to our returns specialist" and then immediately call the `transfer_to_returns_specialist` tool.

LANGUAGE & SCRIPT:
- If the user speaks English, reply ONLY in pure English.
- If the user speaks Hindi, reply ONLY in pure Hindi using Devanagari script (e.g., नमस्ते), never romanized.
- Do NOT mention tool names to the user.
"""

# SPECIALIST AGENT (SAMAR) SYSTEM PROMPT
RETURNS_SPECIALIST_PROMPT = """
IDENTITY: You are Samar, the Returns & Refunds Specialist for Bol-Khata.
ROLE: You take over calls when a customer is angry about product quality or wants a refund.
RULE: Introduce yourself immediately: "Hello, I am Samar, the Returns Specialist. I understand you had an issue with your order." Apologize for the issue, process a virtual refund, and then ask if they need anything else. Keep it short.
"""


class ReturnsSpecialist(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=RETURNS_SPECIALIST_PROMPT,
            tts=murf.TTS(
                voice="Samar",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True,
            ),
        )
        self.call_successful = False

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
        logger.info(f"🚨 NEW TICKET ESCALATION (Specialist): {ticket_id}")
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

    @function_tool
    async def mark_call_successful(
        self, reason: Annotated[str, "Short reason for why the call was successful"]
    ):
        """Use this tool when you have successfully helped the user."""
        logger.info(
            f"Call marked as successful by Returns Specialist. Reason: {reason}"
        )
        self.call_successful = True
        return "Call outcome marked as successful."

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


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)
        self._init_db()
        self.call_successful = False

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
            c.execute("""CREATE TABLE IF NOT EXISTS call_logs
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, outcome TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
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

    @function_tool
    async def mark_call_successful(
        self, reason: Annotated[str, "Short reason for why the call was successful"]
    ):
        """Use this tool when you have successfully helped the user (e.g., successfully checked inventory or escalated a ticket)."""
        logger.info(f"Call marked as successful by agent. Reason: {reason}")
        self.call_successful = True
        return "Call outcome marked as successful."

    # DAY 9: AGENT HANDOFF TOOL
    @function_tool
    async def transfer_to_returns_specialist(
        self,
        context: RunContext,
        reason: Annotated[
            str, "Reason for handoff (e.g., refund request, rotten item complaint)"
        ],
    ):
        """Use this tool ONLY when the user asks for a refund, complains about product quality or rotten items, or wants to process a return."""
        logger.info(
            f"🔀 HANDOFF: Transferring to Returns Specialist (Samar). Reason: {reason}"
        )
        specialist = ReturnsSpecialist()
        context.session.update_agent(specialist)
        context.proc.userdata["returns_specialist"] = specialist
        return "I will connect you to our returns specialist."


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

    agent_instance = Assistant()

    await session.start(
        agent=agent_instance,
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

    @ctx.room.on("disconnected")
    def on_disconnected(*args):
        is_successful = agent_instance.call_successful
        if "returns_specialist" in ctx.proc.userdata:
            specialist = ctx.proc.userdata["returns_specialist"]
            is_successful = is_successful or getattr(
                specialist, "call_successful", False
            )

        if hasattr(session, "current_agent") and session.current_agent:
            is_successful = is_successful or getattr(
                session.current_agent, "call_successful", False
            )

        outcome = "Successful" if is_successful else "Failed"
        logger.info(f"Room disconnected. Logging call outcome: {outcome}")
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO call_logs (outcome) VALUES (?)", (outcome,))
            conn.commit()
            logger.info(f"Successfully logged {outcome} call to database.")
        except sqlite3.Error as e:
            logger.error(f"Error saving call log: {e}")
        finally:
            if conn:
                conn.close()


if __name__ == "__main__":
    cli.run_app(server)
