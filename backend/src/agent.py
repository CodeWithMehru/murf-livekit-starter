import logging
import os

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    room_io,
    tokenize,
)
# IMPORT FIX: Added openai for your Groq setup, and MultilingualModel for the turn detector
from livekit.plugins import murf, silero, deepgram, noise_cancellation, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# STRICT SYSTEM PROMPT: Forces Groq LLM to properly split its brain for English and Hindi/Hinglish
SYSTEM_PROMPT = """
IDENTITY: You are Anisha, a voice assistant for 'Bol-Khata', operating in the Local Commerce track.
OBJECTIVES: Explain how you help street vendors manage their credit (udhaar) ledgers hands-free.
KNOWLEDGE: You only know about your role as a ledger assistant.
LANGUAGE_RULES: 
1. If the user speaks pure English, you MUST reply entirely in English.
2. If the user speaks Hindi or Hinglish, you MUST reply entirely in Hindi/Hinglish.
3. MIRROR the user's language exactly.
GUARDRAILS: 
1. NEVER confirm an order, price, or delivery date.
2. ESCALATION SCRIPT:
   - If user spoke English: "I am sorry, I cannot confirm orders or delivery dates. Please speak to the shopkeeper."
   - If user spoke Hindi/Hinglish: "Maaf kijiye, main order ya delivery date confirm nahi kar sakti. Kripya dukandaar se baat karein."
STYLE: Keep sentences under 15 words. Speak naturally. Never use formatting or emojis.
"""

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # EXACT MURF ANNOUNCEMENT SETTINGS MERGED WITH YOUR GROQ LLM
    session = AgentSession(
        
        # 1. DEEPGRAM (Ears): language="multi" added so it listens to BOTH Hindi and English flawlessly.
        stt=deepgram.STT(model="nova-3", language="multi"),
        
        # 2. LLM (Brain): Your Groq configuration exactly as you wanted (No Gemini).
        llm=openai.LLM(
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY")
        ),
        
        # 3. MURF (Voice): voice="Anisha" (Removed 'hi-IN-') to fix the foreign accent issue exactly as Murf announced.
        tts=murf.TTS(
            voice="Anisha", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        
        # 4. TURN DETECTION: MultilingualModel() added as per Murf's requirement for mixed languages.
        turn_detection=MultilingualModel(),
        
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Start the session
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

    # Join the room and connect to the user
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(server)