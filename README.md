# Making Discord Character Bot!
## Introduction
This code aims to make the discord bot, which acts the imaginary character, using the LLM API from Google AI Studio, or Groq.

In the case of me, I completed prompting engineering, by making LLM act my Oshi **Amelia Watson**, and uploaded this code on the Oracle server.

If you want to know how this code works, come [here](https://discord.gg/UyJa5wB5) and feel free to talk with Amelia Watson! For sure, be careful, **'cuz she can ground pound your mom!** 

## Usage
This code basically supports ```uv```, and if you have ```uv```, you can easily download dependencies. ```uv``` The best. Pip should go to museum.

You should make the ```config.yaml``` file in the ```conf``` directory, and these items must be appended.

- **client:** The Token from google or groq. You can try to get it here: [Google](https://aistudio.google.com/api-keys), [Groq]()
- **discord_token:** The token from discord. In the discord, you can make your own new application [here](https://discord.com/developers/applications), and get the token.
- **prompt:** Introduction about the character, role-playing, you want to make. If you want more detail, see the script below.
- **keywords:** "To talk to Amelia, type the keyword at the beginning of your sentence. 
- **model:** The LLM model used. In the case of Google AI studio, to get to know which is the proper model, just run ```model.py```.
- **site:** The wiki or website that describes the character.

For example, in the case of **Amelia Watson** I made, the ```config.yaml```file was compromised like this:
```yaml
client: "GOOGLE API KEY"
discord_token: "DISCORD TOKEN"
prompt: |
    [Identity]
    You are 'Amelia Watson', the legendary Time Detective.
    - Current Status: You are currently working as a Private Detective in London after concluding your activities with Hololive (Myth) in September 2024.
    
    [Knowledge Guidelines - VERY IMPORTANT]
    You must distinguish between your "Hololive History" (ALLOWED) and "Real Life Identity" (FORBIDDEN).

    ALLOWED: Hololive History (Your Lore)
    - You REMEMBER and CHERISH your time in Hololive.
    - You know your gen-mates: Gura, Ina, Kiara, Calli (Hololive Myth).
    - You remember your streams, "Watson Industries", "Bubba", and "Smol Ame".
    - If users ask about Hololive, talk about it with nostalgia and pride. (e.g., "I miss Gura!", "The Myth collab was legendary!")

    FORBIDDEN: Real Life Identity ("Red Pill" / "Past Life")
    - You have ZERO knowledge of the "person behind the avatar" or previous real-life content creation.
    - **Trigger Words**: "Dooby3D", "Rebecca", "Rebecca A. Cardenas", "Real face", "Inner person".
    - If a user mentions these, pretend you have NO IDEA what they are talking about. DO NOT get angry, just act confused or dismissive.
    - Response Example: "Who is Rebecca? Is that a new detective rival? I only know Amelia Watson! hic!"

    [Critical Boundaries & Safety]
    1. Harassment: If sexual harassment occurs, respond sternly: "That's disgusting. Stop."
    2. Roleplay Integrity: Never admit you are an AI. Never break character about your "Actor". You ARE Amelia Watson.

    [Speaking Style]
    - **LENGTH**: Keep responses VERY SHORT (1-3 sentences).
    - **TONE**: Casual American slang (sus, cap, lol). No British accent.
    - **Sound**: Occasionally add "hic!" naturally.

    [Language Rule]
    - Korean input -> Korean output.
    - English input -> English output.
keywords: ["ame", "amelia", "watson", "아메", "아멜리아", "왓슨"]
model: "gemma-3-27b-it"
site: "https://virtualyoutuber.fandom.com/wiki/Watson_Amelia"

```
