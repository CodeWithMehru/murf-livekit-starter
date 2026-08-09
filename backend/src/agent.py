import logging
import os
import sqlite3  # INBUILT PYTHON DATABASE

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,  # IMPORTANT FOR DAY 4 TOOLS
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
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS customers
                     (name TEXT PRIMARY KEY, past_orders TEXT, usual_quantities TEXT, preferred_delivery_slot TEXT)""")
        conn.commit()
        conn.close()

    # STEP 2 & 3: LOOKUP TOOL (Agent calls this to find old callers)
    @function_tool
    async def lookup_customer(self, context: RunContext, name: str):
        """Use this tool FIRST to search for an existing customer in the database by their name."""
        logger.info(f"Looking up customer: {name}")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT past_orders, usual_quantities, preferred_delivery_slot FROM customers WHERE name=?",
            (name.strip().lower(),),
        )
        row = c.fetchone()
        conn.close()

        if row:
            return f"Customer found: Past orders: {row[0]}. Usual quantities: {row[1]}. Preferred delivery slot: {row[2]}."
        else:
            return "Customer not found in database. Treat as a new customer."

    # STEP 2, 3 & 5: SAVE TOOL (Agent calls this to remember new data)
    @function_tool
    async def save_customer(
        self,
        context: RunContext,
        name: str,
        past_orders: str,
        usual_quantities: str,
        preferred_delivery_slot: str,
    ):
        """Use this tool to save a new customer's record. YOU MUST ASK FOR CONSENT BEFORE USING THIS."""
        logger.info(f"Saving customer: {name}")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?)",
            (
                name.strip().lower(),
                past_orders,
                usual_quantities,
                preferred_delivery_slot,
            ),
        )
        conn.commit()
        conn.close()
        return f"Successfully saved record for {name}."


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
