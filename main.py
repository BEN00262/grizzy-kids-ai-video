# virtual teacher
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

from pydantic import BaseModel, Field
import typing
import os

from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from moviepy.editor import *
from moviepy.video.fx.resize import resize
from moviepy.video.fx.margin import margin
from moviepy.video.VideoClip import TextClip
import azure.cognitiveservices.speech as speechsdk
from moviepy.audio.fx.volumex import volumex
from moviepy.video.fx.mirror_x import mirror_x
from duckduckgo_search import DDGS
# from moviepy.video.fx import crossfadein
import tempfile
from pydub import AudioSegment
from PIL import Image
from rembg import remove
import wget


class Participant(BaseModel):
    name: str = Field(description="name of the participant")
    role: str = Field(description="role of the participant")
    avatar: str = Field(description = "name of the avatar to use")
    gender: str = Field(description="gender of the participant male or female or other")
    voice: str = Field(description="the voice of the participant")

class ConversationPiece(BaseModel):
    speakers_name: str = Field(description="name of the current speaker")
    speaker_voice: str = Field(description="the voice of the current speaker")
    speaker_avatar: str = Field(description = "name of the avatar to use, should be child.png - male, child1.png - female or teacher.png - teacher")
    line: str = Field(description="what the speaker is saying")

class LessonSegmentImage(BaseModel):
    description: str = Field(description="description of the image to be used, should be relevant to the point under discussion")
    placement_indicator: int = Field(description="at what index of the conversation to show the image")
    display_duration: float = Field(description="how long in seconds do we display the image in relation to whats being talked about")

class LessonSegmentMusic(BaseModel):
    description: str = Field(description="description of the music / voice / sound to be used should be relevant to the point under discussion")
    placement_indicator: int = Field(description="at what index of the conversation to play the music / voice / sound")
    mode: str = Field(description="whether the music / voice / sound is a background music / voice / sound or a foreground music / voice / sound")
    volume_level: float = Field(description="the volume level of the music / voice / sound to be played relative to the expected general volume of the podcast")
    play_duration: float = Field(description="how long in seconds do we play the music / voice / sound in relation to whats being talked about")

class LessonSegment(BaseModel):
    name: str = Field(description="name of the segment")
    images: typing.List[LessonSegmentImage] = Field(description="images relevant to the lesson segment")
    sounds_or_music: typing.List[LessonSegmentMusic] = Field(description="music / voice / sounds relevant to the lesson segment")
    conversation: typing.List[ConversationPiece] = Field(description="a list of the conversation segments within the discussion / segment")
    

class Lesson(BaseModel):
    title: str = Field(description="title of the lesson")
    level: str = Field(description="what is the children level ie what grade")
    objectives: typing.List[str] = Field(description="list of objectives that should be achieved at the end of the lesson")

    segments: typing.List[LessonSegment] = Field(description="a list of the lesson segments")


class VirtualTeacherGenerator:
    def __init__(self) -> None:
        self.chat_model = ChatOpenAI(
            openai_api_key = os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo-16k",
            max_tokens=10000,
            temperature=.9
        )
        
        self.parser = PydanticOutputParser(pydantic_object=Lesson)


    def generate_speech_from_text(self, *, voice: str, text: str, audio_file: str):
        """
            supported voices
            -------------------
            	en-GB-SoniaNeural (Female)
                en-GB-RyanNeural (Male)
                en-GB-LibbyNeural (Female)
                en-GB-AbbiNeural (Female)
                en-GB-AlfieNeural (Male)
                en-GB-BellaNeural (Female)
                en-GB-ElliotNeural (Male)
                en-GB-EthanNeural (Male)
                en-GB-HollieNeural (Female)
                en-GB-MaisieNeural (Female, Child)
                en-GB-NoahNeural (Male)
                en-GB-OliverNeural (Male)
                en-GB-OliviaNeural (Female)
                en-GB-ThomasNeural (Male)
        """
        
        speech_config = speechsdk.SpeechConfig(subscription=os.getenv('SPEECH_KEY'), region=os.getenv('SPEECH_REGION'))
        
        audio_config = speechsdk.audio.AudioOutputConfig(
            use_default_speaker = True,
            filename = audio_file
        )

        speech_config.speech_synthesis_voice_name = voice

        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

        speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

        if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized for text [{}], and the audio was saved to [{}]".format(text, audio_file))

        elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_synthesis_result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))

    def generate_image_with_text_and_bubble(self, discussion: ConversationPiece, position: str, direction: str = 'left'):
        SCENE_AUDIO_PATH = os.path.join(f"{position}.wav")

        self.generate_speech_from_text(
            voice = discussion.speaker_voice,
            text = discussion.line,
            audio_file = SCENE_AUDIO_PATH
        )

        scene_audio_segment = AudioFileClip(SCENE_AUDIO_PATH)

        # Load the image
        image_clip = ImageClip(discussion.speaker_avatar, duration = scene_audio_segment.duration)
        image_clip = resize(image_clip, height = 200)
        # image_clip.
        image_clip = margin(image_clip, bottom = 5, left = 5, opacity = 0)
        image_clip = image_clip.set_position(("left", "bottom"))

        if direction == 'right':
            image_clip = margin(image_clip, bottom = 5, right = 5, opacity = 0).set_position(("right", "bottom"))
        else:
            image_clip = margin(image_clip, bottom = 5, left = 5, opacity = 0).set_position(("left", "bottom"))

        # Create a TextClip with the given text
        txt_clip = TextClip(discussion.line, fontsize=40, size=(600, 0), color='black', font="caveatbrush.ttf", method="caption")

        # Calculate the width and height of the text clip
        txt_w, txt_h = txt_clip.size

        chat_bubble = ImageClip("chat.png" if direction == 'left' else "chat_right.png", duration = scene_audio_segment.duration)
        chat_bubble = resize(chat_bubble, height = 1000, width = 1600)

        if direction == 'right':
            chat_bubble = margin(chat_bubble, top = 30, right = 50, opacity = 0).set_position(("right", "top"))
        else:
            chat_bubble = margin(chat_bubble, top = 30, left = 50, opacity = 0).set_position(("left", "top"))

        bubble_width, bubble_height = chat_bubble.size

        x_position = ((bubble_width - txt_w) / 2) + (0 if direction == 'left' else 650)
        y_position = ((bubble_height - txt_h) / 2) - 100

        txt_clip = txt_clip.set_position((x_position, y_position))

        # Set the duration of the text clip to match the image duration
        txt_clip = txt_clip.set_duration(scene_audio_segment.duration)
        chat_bubble = chat_bubble.set_duration(scene_audio_segment.duration)

        # Overlay the bubble, text, and image
        final_clip = CompositeVideoClip([chat_bubble, txt_clip, image_clip], size = (1920, 1080))

        final_clip = final_clip.set_duration(scene_audio_segment.duration)
        final_clip = final_clip.set_audio(scene_audio_segment)        

        return final_clip
    
    def download_relevant_images_gifs(self, *, file_path: str, description: str, gif: bool = False):
        """
        used to download resources from the internet to be used to create a video
        """
        with DDGS() as ddgs:
            # keywords = 'Image of solar panels animated'
            ddgs_images_gen = ddgs.images(
                keywords = f"{description} animated",
                region="us-en",
                safesearch="off",
                size="Medium",
                color="color",
                type_image = "gif" if gif else "transparent",
                layout="Square",
                license_image=None,
            )

            for r in ddgs_images_gen:
                try:
                    path = wget.download(r["image"], file_path)

                    if path is not None:
                        image = Image.open(path)
                        image = remove(image.resize(size=(400, 300)))
                        image.save(file_path)

                        return file_path
                except Exception as e:
                    print(e)

        raise Exception("Failed to download a relevant image")


    def generate_lesson(self, *, title: str, participants: typing.List[Participant]):
        prompt = PromptTemplate(
            template="You are an experienced teacher with years of experience under your belt work as an illustrative lessons creator, create a lesson plan with the title {title}. The plan should have the following participants \n\n\n{participants}.\n\n\n The lesson plan should be illustrative enough to be engaging to students who will be watching it. Use images, sounds, voices, stories and analogies to further spruce up the lesson plan. the format of the lesson plan should as a recordable artefact.\n{format_instructions}\n",
            input_variables=["title"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions(),
                "participants": ", ".join([f"{p.name} ({p.gender}, {p.role} with a voice {p.voice})" for p in participants])
            }
        )

        messages = [
            HumanMessage(
                content = prompt.format_prompt(title = title).to_string()
            )
        ]

        result = self.parser.parse(self.chat_model.generate(messages=[messages]).generations[0][0].text)

        print(result)

        with tempfile.TemporaryDirectory() as tmpdirname:
            DURATION = 2 # seconds

            TITLE_AUDIO_PATH = os.path.join(tmpdirname, "title.wav")

            self.generate_speech_from_text(
                voice = "en-GB-SoniaNeural",
                text = f"Hello, welcome today we are going to learn about the '{result.title}'",
                audio_file = TITLE_AUDIO_PATH
            )

            OBJECTIVES_AUDIO_PATH = os.path.join(tmpdirname, "objectives.wav")

            lessons_to_be_learnt = ".\n".join(result.objectives)

            self.generate_speech_from_text(
                voice = "en-GB-SoniaNeural",
                text = f"By the end of the lesson you will learn about.\n\n\n{lessons_to_be_learnt}",
                audio_file = OBJECTIVES_AUDIO_PATH
            )

            title_audio_segment = AudioFileClip(TITLE_AUDIO_PATH)

            title_scene = TextClip(result.title, fontsize=70, color='black', bg_color='white', font="caveatbrush.ttf", method="caption", size=(600, 0))
            title_scene = title_scene.set_position('top').set_duration(title_audio_segment.duration)
            title_scene = title_scene.set_audio(title_audio_segment)
            

            objectives_audio_segment = AudioFileClip(OBJECTIVES_AUDIO_PATH)

            objectives_scene = TextClip("\n\n".join(result.objectives), fontsize=40, color='black', bg_color='white', method="caption", size=(600, 0), font="caveatbrush.ttf")
            objectives_scene = objectives_scene.set_position('center').set_duration(objectives_audio_segment.duration)
            objectives_scene = objectives_scene.set_audio(objectives_audio_segment)

            scenes = [title_scene, objectives_scene]

            for segment_position, segment in enumerate(result.segments):
                # get the segment images
                # save the first one then try to place them in a frame

                segment_scene = TextClip(segment.name, fontsize=30, color='black', bg_color='white', font="caveatbrush.ttf", method="caption", size=(600, 0))
                segment_scene = segment_scene.set_position('top').set_duration(DURATION)
                scenes.append(segment_scene)

                segment_scene_sub_timeline = [None] * (len(segment.images) + len(segment.conversation))

                # download all the images first
                # this might take sometime :)
                # we should parallelize this bit btw -- but who cares
                for position, image in enumerate(segment.images):
                    try:
                        image_path = self.download_relevant_images_gifs(
                            file_path = os.path.join(tmpdirname, f"image-{segment_position}-{position}"),
                            description = image.description
                        )

                        # show an image
                        # start at 5 seconds -- fix that algo later
                        downloaded_image = ImageClip(image_path, duration = DURATION)
                        downloaded_image = downloaded_image.set_position(("center", "center"))

                        # constraint the index not to be beyond or below the list
                        segment_scene_sub_timeline[image.placement_indicator % len(segment_scene_sub_timeline)] = downloaded_image

                        # should we describe the image -- thats a job for the weekend

                    except Exception as e:
                        # ignore this exception -- we dont care 
                        pass

                for position, discussion in enumerate(segment.conversation):
                    clip = self.generate_image_with_text_and_bubble(
                        discussion,
                        f"{segment_position}_{position}",
                        'right' if position % 2 == 0 else 'left' 
                    )

                    # find a relevant position in the sub scene timeline
                    if segment_scene_sub_timeline[position] is not None:
                        # start at my current position and try to find an empty spot for me
                        local_position = position

                        while local_position < len(segment_scene_sub_timeline):
                            if segment_scene_sub_timeline[local_position] is None:
                                segment_scene_sub_timeline[local_position] = clip
                                break

                            local_position += 1

                        # throw an execption
                        raise Exception("clip overflow in the current scene")
                    else:
                        segment_scene_sub_timeline[position] = clip

                scenes.append(*filter(lambda x: x is not None, segment_scene_sub_timeline))

            video_clip = concatenate_videoclips(scenes, method="compose")

            background_music = AudioFileClip("background.mp3")

            video_clip = video_clip.set_audio(
                CompositeAudioClip([
                    volumex(background_music.set_duration(video_clip.duration), 0.1),
                    video_clip.audio
                ])
            )

            video_clip = resize(video_clip, height = 1080, width = 1920)

            background_image = ImageClip("background.jpg")  # Replace with your image file
            background_image = background_image.resize(video_clip.size)

            video_clip = CompositeVideoClip([
                background_image.set_duration(video_clip.duration),
                video_clip.set_duration(video_clip.duration)
            ])

            video_clip.write_videofile(f"{title}.mp4", fps=24, audio_codec="aac", audio_bitrate="192k")

if __name__ == "__main__":
    virtual = VirtualTeacherGenerator()

    virtual.generate_lesson(
        title = "Climate change",
        participants = [
            Participant(name="Mr. Juma", role="teacher", gender="female", voice="en-GB-RyanNeural", avatar="teacher.png"),
            Participant(name="Brian", role="learner", gender="male", voice="en-US-AnaNeural", avatar="child.png"),
            Participant(name="Emma", role="learner", gender="female", voice="en-GB-MaisieNeural", avatar="child1.png")
        ]
    )