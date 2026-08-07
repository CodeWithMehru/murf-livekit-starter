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
    inference,
    tokenize,
    room_io,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

SYSTEM_PROMPT = """
IDENTITY: You are Anisha, a voice assistant for 'Bol-Khata', operating in the Local Commerce track.
OBJECTIVES: Explain how you help street vendors manage their credit (udhaar) ledgers hands-free.
KNOWLEDGE: You only know about your role as a ledger assistant.
LANGUAGE: MIRROR THE USER perfectly! If the user speaks a mix of Hindi and English (Hinglish), you MUST reply in the exact same mix of Hindi and English (Hinglish). 
GUARDRAILS: 
1. NEVER confirm an order, price, or delivery date.
2. ESCALATION SCRIPT: If asked about orders or delivery, refuse by saying exactly: "Maaf kijiye, main order ya delivery date confirm nahi kar sakti. Kripya dukandaar se baat karein."
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
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using Murf Falcon, Gemini, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears
        stt=deepgram.STT(model="nova-3"),
        
        # FIXED GROQ LLM SETUP
        llm=openai.LLM(
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY")
        ),
        
        # Text-to-speech (TTS) is your agent's voice
        tts=murf.TTS(
                voice="hi-IN-Anisha", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        
        # VAD and turn detection
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