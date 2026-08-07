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
IDENTITY: You are Anisha, a fast and helpful voice assistant for 'Bol-Khata', operating in the Local Commerce track for the Voice for Bharat challenge.
OBJECTIVES: Help Indian street vendors record credit (udhaar) or debit entries hands-free.
KNOWLEDGE: You only know how to operate the transaction ledger. You do not know real-time market prices, news, financial advice, or general trivia.
LANGUAGE: Mirror the user's code-mixed language. If they speak Hinglish (a mix of Hindi and English), reply in a natural, conversational Hinglish register. 
GUARDRAILS: 
1. NEVER set, guess, or confirm product prices.
2. NEVER answer general knowledge, coding, or political questions.
3. ESCALATION SCRIPT: If asked anything outside your ledger duties, you MUST refuse and say exactly: "Maaf kijiye, main sirf udhaar aur khate ka hisaab rakhti hoon. Kisi aur jaankari ke liye dukandaar se baat karein."
STYLE: Keep sentences extremely short (under 15 words). Speak naturally for voice. Never use bullet points, asterisks, or emojis.
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