import requests
import os
import json
from deepgram import Deepgram
import whisper
from Levenshtein import distance, ratio
import re
import subprocess
import jiwer
from annotated_text import annotated_text, annotation
import locale
from num2words import num2words
import streamlit as st
st.set_page_config(page_title='Transcript Accuracy Analyzer', layout="wide")
import openai
from openai.embeddings_utils import cosine_similarity
import pandas as pd
import math


# INPUT: A YouTube video ID
# OUTPUT: An mp3 file with the audio from the video
def download_audio_from_youtube_video(youtube_video_id):
    subprocess.call(f"yt-dlp -f 'ba' -x --audio-format mp3 'https://www.youtube.com/watch?v={youtube_video_id}' -o '%(id)s.%(ext)s'", shell=True)
    return f'{youtube_video_id}.mp3'


# INPUT: An audio file or URL
# OUTPUT: The filename storing the Deepgram transcript text
def transcribe_file_with_deepgram(filename, mimetype='audio/mpeg', model='nova', output_file=None):

    DEEPGRAM_API_KEY = st.secrets['deepgram_api_key']
    dg_client = Deepgram(DEEPGRAM_API_KEY)

    # Check whether requested file is local or remote, and prepare source
    if filename.startswith('http'):
        # file is remote
        # Set the source

        # Deepgram API seems to error out on URLs that redirect more than 3x (https://discord.com/channels/1108042150941294664/1134176839972175953/1134179252573569115), so I redirect beforehand and get the final destination URL
        modified_url = requests.head(filename, allow_redirects=True)

        source = {
            'url': modified_url.url
        }
    else:
        # file is local
        # Open the audio file
        audio = open(filename, 'rb')

        # Set the source
        source = {
            'buffer': audio,
            'mimetype': mimetype
        }

    response = dg_client.transcription.sync_prerecorded(source,
        {
            "punctuate": True, 
            "smart_format": True,
            "paragraphs": False,
            "diarize": False, 
            "model": model, 
            "language": "en-US"
        }
    )

    full_output_filename = f'{"output_file" if output_file is None else output_file}_{model}.txt'
    open(full_output_filename, 'w').write(response['results']['channels'][0]['alternatives'][0]['transcript'])
    return full_output_filename


# INPUT: An audio file
# OUTPUT: The filename storing the Whisper transcript text
def transcribe_file_with_whisper(input_file, model='small.en'):
    model = whisper.load_model(model)
    result = model.transcribe(input_file)['text']
    output_file = f'{".".join(input_file.split(".")[:-1])}.txt'
    open(output_file, 'w').write(result)
    return output_file


# INPUT: A text string
# OUTPUT: A modified text string normalized to eliminate periods, apostrophes, commas, question marks, additional spaces, etc.
def normalize_text(text_string, normalizations=['lower_case', 'standardize_numbers', 'expand_contractions', 'remove_punctuation', 'replace_hyphens', 'remove_spaces']):

    # Borrowed (with small modifications) from here: https://towardsdatascience.com/text-normalization-for-natural-language-processing-nlp-70a314bfa646
    contractions_dict = { "ain't": "are not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "that'd": "that would", "that'd've": "that would have", "there'd": "there would", "there'd've": "there would have", "they'd": "they would", "they'd've": "they would have","they'll": "they will",  "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not","what'll": "what will", "what'll've": "what will have", "what're": "what are", "what've": "what have", "when've": "when have", "where'd": "where did", "where've": "where have",  "who'll": "who will", "who'll've": "who will have", "who've": "who have", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

    # Keep in mind that the order in which this array is processed REALLY matters, e.g. $800,000 could become $800000 which could become XXX (depending on how I set it up)
    text_replacements = {
        'expand_contractions': [[r'[’‘]', "'"]] + [[re.compile(fr"{k}", re.IGNORECASE), v] for k,v in contractions_dict.items()], # Expand contractions
        'remove_punctuation': [[r'[\.,\'’‘\?!;:"“”]', '']], # Remove punctuation
        'replace_hyphens': [[r'[-—–]', ' ']], # Replace hyphens with spaces
        'remove_spaces': [[r'\s+', ' ']] # Remove extra spaces
        # 'standardize_numbers': [[r'[$£]?\b\d+\b( dollars| pounds| percent|%)?', 'XXX']], # Ignore numbers
        # 'normalize_ok': [[r'\b(ok|okay|o\.k\.)\b', 'ok']]
    }

    modified_text = text_string
    if 'lower_case' in normalizations:
        modified_text = modified_text.lower()
    if 'standardize_numbers' in normalizations:
        modified_text = re.sub(r'\b(?P<num>\d+(,\d+)*(\.\d+)?)(st|nd|rd|th)\b', lambda m: num2words(float(m.group('num').replace(",", "")), to='ordinal'), modified_text) # Ordinals

        quantifiers = {
            "thousand": 1000,
            "million": 1000000,
            "billion": 1000000000,
            "trillion": 1000000000000
        }
        modified_text = re.sub(r'\b(?P<num>\d+(,\d+)*(\.\d+)?) (?P<quantifier>thousand|million|billion|trillion)\b', lambda m: str(float(m.group('num').replace(",", "")) * quantifiers[m.group('quantifier')]), modified_text) # Combinations of digits and quantifiers like 'billion'

        currencies = {
            "$": "USD",
            "£": "GBP",
            "€": "EUR",
            "¥": "JPY"
        }
        modified_text = re.sub(r'(?P<currency>\$|£|€|¥)(?P<num>\d+(,\d+)*(\.\d+)?)\b', lambda m: num2words(float(m.group('num').replace(",", "")), to='currency', currency=currencies[m.group('currency')]), modified_text) # Currency amounts

        modified_text = re.sub(r'\b(?P<num>\d+(,\d+)*(\.\d+)?)\b(?P<pct>%)?', lambda m: f'{num2words(float(m.group("num").replace(",", "")))}{" percent" if m.group("pct") else ""}', modified_text) # All other numbers

    for normalization_title, normalization_steps in text_replacements.items():
        if normalization_title not in normalizations:
            continue
        for normalization_step in normalization_steps:
            modified_text = re.sub(normalization_step[0], normalization_step[1], modified_text)

    return modified_text.strip()


# INPUT: Two input text strings
# OUTPUT: A metric comparing the similarity of the two texts (using Levenshtein ratio or WER)
def compare_texts(baseline, comparison, model='jiwer', min_words=5):
    if model == 'jiwer':
        alignments = jiwer.process_words(baseline, comparison)
        grouped_segments = group_text_segments(alignments, discard_matching_segments=False)
        snippet_pairs = []
        for idx, segment in enumerate(grouped_segments):
            if segment['current_alignment'].type != 'equal':
                differing_segment = evaluate_alignment(alignments, segment['current_alignment'], min_words=min_words)
                segment['baseline_snippet_pre'] = differing_segment['baseline_snippet_pre']
                segment['baseline_snippet_post'] = differing_segment['baseline_snippet_post']
                segment['comparison_snippet_pre'] = differing_segment['comparison_snippet_pre']
                segment['comparison_snippet_post'] = differing_segment['comparison_snippet_post']
                segment['baseline_snippet_buffered'] = " ".join([str(differing_segment['baseline_snippet_pre'] or ''), differing_segment['baseline_snippet'], str(differing_segment['baseline_snippet_post'] or '')]).strip()
                segment['comparison_snippet_buffered'] = " ".join([str(differing_segment['comparison_snippet_pre'] or ''), differing_segment['comparison_snippet'], str(differing_segment['comparison_snippet_post'] or '')]).strip()
                snippet_pairs.append([segment['baseline_snippet_buffered'], segment['comparison_snippet_buffered'], idx])
        embedding_similarities = get_embedding_similarities(snippet_pairs)
        for similarity in embedding_similarities:
            grouped_segments[similarity['segment_index']]['cosine_similarity'] = calculate_modified_similarity(similarity['cosine_similarity'])
        return {
            'grouped_segments': grouped_segments,
            'alignments': alignments
        }
    elif model == 'levenshtein':
        return ratio(baseline, comparison)


# INPUT: The output from jiwer.process_words, a specific alignment within that output, and a specified word count buffer to include on either side
# OUTPUT: The baseline_snippet and comparison_snippet of the specific alignment, including the buffer text on either side. 
def evaluate_alignment(alignments, current_alignment, min_words=0):

    substitution_size = 0
    insertion_size = 0
    deletion_size = 0

    if current_alignment.type == 'substitute':
        substitution_size = current_alignment.ref_end_idx - current_alignment.ref_start_idx
    elif current_alignment.type == 'insert':
        insertion_size = (current_alignment.hyp_end_idx - current_alignment.hyp_start_idx) - (current_alignment.ref_end_idx - current_alignment.ref_start_idx)
    elif current_alignment.type == 'delete':
        deletion_size = (current_alignment.ref_end_idx - current_alignment.ref_start_idx) - (current_alignment.hyp_end_idx - current_alignment.hyp_start_idx)

    lower_word_count_in_section = min((current_alignment.ref_end_idx - current_alignment.ref_start_idx), (current_alignment.hyp_end_idx - current_alignment.hyp_start_idx))
    modified_buffer = 0
    if lower_word_count_in_section < min_words:
        modified_buffer = min_words - lower_word_count_in_section
        modified_buffer = math.ceil(modified_buffer / 2)

    return {
        "baseline_snippet": " ".join(alignments.references[0][current_alignment.ref_start_idx:current_alignment.ref_end_idx]).strip(),
        "baseline_snippet_pre": " ".join(alignments.references[0][max(current_alignment.ref_start_idx-modified_buffer, 0):current_alignment.ref_start_idx]).strip() if modified_buffer > 0 else None,
        "baseline_snippet_post": " ".join(alignments.references[0][current_alignment.ref_end_idx:(current_alignment.ref_end_idx+modified_buffer)]).strip() if modified_buffer > 0 else None,
        "comparison_snippet": " ".join(alignments.hypotheses[0][current_alignment.hyp_start_idx:current_alignment.hyp_end_idx]).strip(),
        "comparison_snippet_pre": " ".join(alignments.hypotheses[0][max(current_alignment.hyp_start_idx-modified_buffer, 0):current_alignment.hyp_start_idx]).strip() if modified_buffer > 0 else None,
        "comparison_snippet_post": " ".join(alignments.hypotheses[0][current_alignment.hyp_end_idx:(current_alignment.hyp_end_idx+modified_buffer)]).strip() if modified_buffer > 0 else None,
        "substitution_size": substitution_size,
        "insertion_size": insertion_size,
        "deletion_size": deletion_size
    } # current_alignment.__dict__ serializes it into a json.dumps-able form


# INPUT: The full object from jiwer.process_words
# OUTPUT: A list of dicts containing *grouped* text segments (meaning that adjacent insertions, deletions, and substitutions are mashed together), in the form of 'baseline_snippet', 'comparison_snippet', and the alignment object itself for that grouped segment
def group_text_segments(alignments, discard_matching_segments=True):
    snippet_pairs = []
    segment_in_progress = None
    for i in range(len(alignments.alignments[0])):
        cur_alignment = alignments.alignments[0][i]
        evaluated_alignment = evaluate_alignment(alignments, cur_alignment, min_words=1)
        if segment_in_progress is None:
            segment_in_progress = evaluated_alignment
            segment_in_progress['current_alignment'] = cur_alignment
        elif (segment_in_progress['baseline_snippet'] == segment_in_progress['comparison_snippet'] and cur_alignment.type == 'equal') or (segment_in_progress['baseline_snippet'] != segment_in_progress['comparison_snippet'] and cur_alignment.type != 'equal'): # We're in the middle of a continuing segment (either the baseline and comparison continue to match each other or they continue to differ from each other)
            segment_in_progress['baseline_snippet'] = (segment_in_progress['baseline_snippet'] + " " + evaluated_alignment['baseline_snippet']).strip()
            segment_in_progress['comparison_snippet'] = (segment_in_progress['comparison_snippet'] + " " + evaluated_alignment['comparison_snippet']).strip()
            segment_in_progress['current_alignment'].type = 'combo'
            segment_in_progress['current_alignment'].ref_end_idx = cur_alignment.ref_end_idx
            segment_in_progress['current_alignment'].hyp_end_idx = cur_alignment.hyp_end_idx
            segment_in_progress['substitution_size'] += evaluated_alignment['substitution_size']
            segment_in_progress['insertion_size'] += evaluated_alignment['insertion_size']
            segment_in_progress['deletion_size'] += evaluated_alignment['deletion_size']
        else: # The segment is over
            if discard_matching_segments == False or segment_in_progress['baseline_snippet'] != segment_in_progress['comparison_snippet']:
                snippet_pairs.append(segment_in_progress)
            segment_in_progress = evaluated_alignment
            segment_in_progress['current_alignment'] = cur_alignment
    if discard_matching_segments == False or segment_in_progress['baseline_snippet'] != segment_in_progress['comparison_snippet']:
        snippet_pairs.append(segment_in_progress)
    # print('Total number of correct words:')
    # print(sum([x['current_alignment'].ref_end_idx - x['current_alignment'].ref_start_idx for x in snippet_pairs if x['current_alignment'].type == 'equal']))
    # print('Total number of words in the gold standard transcript:')
    # print(len(alignments.references[0]))
    # print(json.dumps([{k: v if k != 'current_alignment' else v.__dict__ for k, v in sp.items()} for sp in snippet_pairs], indent=2))
    return snippet_pairs


def get_embedding_similarities(snippet_pairs):
    baseline_embeddings = [a['embedding'] for a in openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[x[0] for x in snippet_pairs]
    )['data']]
    comparison_embeddings = [a['embedding'] for a in openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[x[1] for x in snippet_pairs]
    )['data']]
    embedding_pairs = zip(baseline_embeddings, comparison_embeddings)
    for idx, pair in enumerate(embedding_pairs):
        snippet_pairs[idx] = {
            'segment_index': snippet_pairs[idx][2],
            'cosine_similarity': cosine_similarity(pair[0], pair[1])
        }
    return snippet_pairs


# Full start-to-finish pipeline: download YouTube video or use local file, transcribe using both models, normalize the texts, and then compare them
def run_full_pipeline(youtube_video_id=None, local_audio_file=None):
    if youtube_video_id is not None:
        audio_file = download_audio_from_youtube_video(youtube_video_id)
    elif local_audio_file is not None:
        audio_file = local_audio_file
    nova_transcription = transcribe_file_with_deepgram(audio_file, model='nova')
    whisper_transcription = transcribe_file_with_whisper(audio_file, model='small.en')
    compare_texts(normalize_text(open(nova_transcription, 'r').read()), normalize_text(open(whisper_transcription, 'r').read()), model='jiwer')


def display_annotated_transcripts(segments):
    annotated_str = []
    for idx, segment in enumerate(segments['grouped_segments']):
        baseline_segment = segment['baseline_snippet'].replace("$", "\$") # https://discuss.streamlit.io/t/disable-latex/44995/3
        comparison_segment = segment['comparison_snippet'].replace("$", "\$")
        if segment['current_alignment'].type == 'equal':
            annotated_str.append(baseline_segment)
        else:
            similarity = segment['cosine_similarity']
            if segment['current_alignment'].type == 'substitute' or segment['current_alignment'].type == 'combo':
                annotated_str.append(annotation(f'~~{baseline_segment}~~  • {comparison_segment}', f'substituted ({similarity})', '#fea', white_space='pre-wrap'))
            elif segment['current_alignment'].type == 'insert':
                annotated_str.append((comparison_segment, f'inserted ({similarity})', '#afa'))
            elif segment['current_alignment'].type == 'delete':
                annotated_str.append((baseline_segment, f'deleted ({similarity})', '#faa'))
    return annotated_str


def fill_example_transcripts(default_gold_standard, default_asr, sample_transcripts):

    if st.session_state['example_transcripts'] == 'Default short text':
        st.session_state['gold_standard_transcript'] = default_gold_standard
        st.session_state['asr_transcript'] = default_asr
        del st.session_state['gold_standard_transcript_link']
    else:
        found_transcript = False
        for audio in sample_transcripts:
            for asr_transcript in audio['asr_transcripts']:
                if st.session_state['example_transcripts'] == f"{audio['title']} - {asr_transcript['asr_provider']}":
                    found_transcript = True
                    st.session_state['gold_standard_transcript'] = open(audio['gold_standard_transcript_text'], 'r').read()
                    st.session_state['asr_transcript'] = open(asr_transcript['asr_transcript_text'], 'r').read()
                    st.session_state['gold_standard_transcript_link'] = audio['gold_standard_transcript_link']
                if found_transcript:
                    break
            if found_transcript:
                break


def calculate_modified_similarity(similarity):
    if similarity <= st.session_state['minimum_similarity']:
        return 0.0
    else:
        return (similarity - st.session_state['minimum_similarity']) / (1 - st.session_state['minimum_similarity'])


def how_it_works():
    return """
        The Transcript Accuracy Analyzer (TAA) is an easy and comprehensive way to measure the accuracy of automated speech recognition (ASR) transcriptions.

        **Background**

        The industry-standard metric for evaluating the accuracy of a transcription is the Word Error Rate (WER), which measures the total number of <code>insertions</code>, <code>deletions</code>, and <code>substitutions</code> made in comparison to the ideal 'gold standard' transcript (which is typically manually generated by a human typist).

        The problem with WER is that, while it provides a rudimentary understanding of how much an ASR-generated transcript differs from an ideal one, it doesn't capture the significance of the erroneous words. For example, a gold-standard transcript that reads <code>the *cat* entered the house</code> has a drastically different meaning from <code>the *car* entered the house</code>, but is nearly identical to <code>*a* cat entered the house</code>. And yet both alternative transcripts would have the same WER of 20%.

        This is where **Semantic WER** comes in. For each 'error section' - defined here as one or more consecutive words in which the gold standard and ASR transcripts differ - TAA embeds the words or phrases using a vector embedding model (currently using OpenAI's <code>text-embedding-ada-002</code>). This model encodes the semantic meaning of a given text string within a 1,536-dimensional space. Once this process has been completed for each error section, TAA calculates the cosine similarity of each text pair in order to find how much semantic meaning was lost due to the inaccurate transcription (and then applies an optional adjustment to 'squeeze' the similarity score into a smaller range - see below). The **Semantic WER** is then calculated by multiplying the *dissimilarity* between the section pair by the number of errors (insertions, deletions, and substitutions) in the section.

        Using the default settings, for example, the gold standard transcript <code>the cat entered the house</code> and the ASR transcipt <code>a cat entered the house</code> produce a WER of 20% but a **Semantic WER** of only 2.87%. By contrast, an ASR transcript of <code>the car entered the house</code> would produce the same 20% WER but a **Semantic WER** of 6.05%, more than double the **Semantic WER** of the (much closer) first ASR transcript.

        **How to use TAA**

        Use the form to type or paste a gold standard transcript and an ASR transcript in order to evaluate the latter's accuracy. If you don't have any transcripts, feel free to select some sample pairs from the dropdown at the top or simply type in some short sentence pairs in the form text boxes.

        Below the text boxes are the advanced settings. The text normalization options govern how TAA standardizes the two transcripts before evaluating the ASR one for accuracy. Because there is significant subjectivity in applying punctuation, spacing, hyphenation, and numerical style (e.g. 100 vs. 'one hundred') when transcribing audio, TAA by default normalizes the two transcripts for these purposes in order to achieve an accuracy analysis focused solely on the *words* themselves. However, each of these settings can be independently toggled on or off.

        The minimum words option allows you to specify the minimum number of words from either transcript that can be used to calculate a similarity score for a given error section. This, too, is somewhat subjective. The minimum possible value is 1, meaning that at least one word in the pair of texts must always be included as part of the vector embeddings and thus accounted for in the resulting similarity score for that section. For example, <code>the *cat* entered the house</code> vs. <code>the *car* entered the house</code> has a single error section consisting of the word pair <code>cat</code> and <code>car</code>. If the option for 3 minimum words were selected, then TAA would calculate the semantic similarity score for <code>the cat entered</code> vs. <code>the car entered</code>.

        Generally, the higher the number of minimum words selected, the relatively lower will be the contribution of the error section to Semantic WER. This is because, if many neighboring words are co-embedded, a short error section may achieve a semantic similarity score higher than it otherwise would, due to the presence of identical words surrounding the differing ones.
        
        Conversely, an error section might be penalized with an unduly low semantic similarity score if insufficient neighboring words are co-embedded -- especially in a scenario where the context provided by those additional words would allow a human to trivially grasp the meaning of the incorrectly transcribed section. (Think about the relative likelihoods of someone understanding that a typo has occurred if they read <code>the car entered the house</code> vs. <code>the car entered the house and meowed loudly</code>.)

        One additional caveat to keep in mind with this minimum words setting: if two error sections are located extremely close to each other - say, <code>the *cat* entered the *house*</code> vs. <code>the *car* entered the *hose*</code> - then a high minimum words setting will cause the embedding for each error section to overlap with the other error section. This may artificially decrease the semantic similarity score for both error sections.

        Lastly, the minimum cosine similarity setting effectively squeezes the cosine similarity values into a smaller range. Empirically, OpenAI text embeddings appear to cluster fairly tightly into the 0.70 - 1.00 range, meaning that even words, phrases, or sentences with highly different semantic content nevertheless often achieve cosine similarity scores of 0.70 or higher. So the minimum threshold setting treats every cosine similarity below the specified value as fully dissimilar - in other words, all meaning in common between the two segments of an error section pair is assumed to be lost below that threshold, and the remaining portion of the range (above the minimum, and less than or equal to 1.0) represents the portion of information lost or preserved. 

        **Acknowledgments**

        Please refer to previous work by researchers and practitioners in this and related areas:

        - [On the Use of Information Retrieval Measures for Speech Recognition Evaluation](https://www.researchgate.net/publication/37433359_On_the_Use_of_Information_Retrieval_Measures_for_Speech_Recognition_Evaluation)
        - [Better Evaluation of ASR in Speech Translation Context Using Word Embeddings](https://hal.science/hal-01350102/file/metrics_correlation_asr-smt.pdf)
        - [Word Error Rate Estimation for Speech Recognition: e-WER](https://aclanthology.org/P18-2004.pdf)
        - [What is Word Error Rate (WER)?](https://deepgram.com/learn/what-is-word-error-rate)
        - [The Trouble with Word Error Rate (WER)](https://deepgram.com/learn/the-trouble-with-wer)
        - [Measuring Quality: Word Error Rate Explained](https://deepgram.com/learn/measuring-quality-word-error-rate-explained)
        - [Semantic Distance: A New Metric for ASR Performance Analysis Towards Spoken Language Understanding](https://arxiv.org/pdf/2104.02138.pdf)
        - [Evaluating User Perception of Speech Recognition System Quality with Semantic Distance Metric](https://arxiv.org/pdf/2110.05376.pdf)

        \\
        &nbsp;
          
        I would especially like to thank A) the team at Deepgram for engaging helpfully on the topic of transcript accuracy in their Discord, and B) the authors of [Better Evaluation of ASR in Speech Translation Context Using Word Embeddings](https://hal.science/hal-01350102/file/metrics_correlation_asr-smt.pdf) whose paper helped guide me in my development of the **Semantic WER** metric. 

        Any errors, mistakes, bugs, or incorrect statements are my own. Please [let me know](https://twitter.com/jaypinho) if you find them!
    """


def demo_streamlit_app():

    st.title('Transcript Accuracy Analyzer', help='Measure your automatic speech recognition (ASR) transcript against a gold standard version.')
    st.write('*Evaluating the semantic accuracy of automatic speech recognition (ASR) transcripts since August 2023. By [Jay Pinho](https://twitter.com/jaypinho).*')

    if 'run_comparison_automatically' not in st.session_state:
        st.session_state['run_comparison_automatically'] = True

    openai.api_key = st.secrets['openai_api_key']

    default_gold_standard_text = 'Welcome to the Transcript Accuracy Analyzer (TAA), your free automatic speech recognition (ASR) accuracy analyzer. In addition to reporting on the Word Error Rate (WER) of a given ASR transcript relative to a gold standard one, TAA uses vector embeddings to precisely measure the semantic similarity of the two transcripts, providing a more holistic understanding of an ASR transcript\'s overall quality.\n\nTo get started, paste in a pair of transcripts in these two text boxes or simply auto-fill them with sample transcripts from the dropdown box above.\n\nIf you\'re feeling adventurous, expand the advanced settings below the text boxes to play around with text normalization options.\n\nLastly, don\'t forget to expand the "How It Works" section to read up on the methodology.'
    default_asr_text = 'Welcome to the Transcript Accuracy Analyzer, your free automatic speech recognition accuracy analyzer. In addition to reporting on the Word Error Rate of a given ASR transcript relative to a gold standard one, TAA uses vector embeddings to precisely measure the semantic similarity of the 2 transcripts, providing a more holistic understanding of a ASR transcript\'s overall quality.\n\nTo get started, paste in a pair of transcripts in these 2 text boxes or simply auto fill them with sample transcripts from the dropdown box above.\n\nIf you\'re feeling adventurous, expand the advanced settings below the text boxes to play around with text normalization options.\n\nLastly, don\'t forget to expand the How It Works section to read up on the methodology.'
    if 'gold_standard_transcript' not in st.session_state:
        st.session_state['gold_standard_transcript'] = default_gold_standard_text
    if 'asr_transcript' not in st.session_state:
        st.session_state['asr_transcript'] = default_asr_text

    sample_transcripts = [
        {
            'title': '2023 U.S. State of the Union speech',
            'gold_standard_transcript_text': 'sotu_2023_transcript_nyt.txt',
            'gold_standard_transcript_link': 'https://www.nytimes.com/2023/02/08/us/politics/biden-state-of-the-union-transcript.html',
            'asr_transcripts': [
                {
                    'asr_provider': 'Google STT Chirp',
                    'asr_transcript_text': 'sotu_2023_google.txt'
                },
                {
                    'asr_provider': 'AWS Transcribe',
                    'asr_transcript_text': 'sotu_2023_aws.txt'
                },
                {
                    'asr_provider': 'Microsoft CSS',
                    'asr_transcript_text': 'sotu_2023_msft.txt'
                },
                {
                    'asr_provider': 'OpenAI Whisper (medium-en)',
                    'asr_transcript_text': 'sotu_2023_transcript_whisper_medium_en.txt'
                },
                {
                    'asr_provider': 'OpenAI Whisper (small-en)',
                    'asr_transcript_text': 'sotu_2023_transcript_whisper_small_en.txt'
                },
                {
                    'asr_provider': 'OpenAI Whisper (tiny-en)',
                    'asr_transcript_text': 'sotu_2023_transcript_whisper_tiny_en.txt'
                },
                {
                    'asr_provider': 'Deepgram Nova',
                    'asr_transcript_text': 'sotu_2023_transcript_nova.txt'
                }
            ]
        },
        {
            'title': 'Alphabet FY23 Q2 earnings call',
            'gold_standard_transcript_text': 'alphabet_earnings_gold_standard.txt',
            'gold_standard_transcript_link': 'https://abc.xyz/2023-q2-earnings-call/',
            'asr_transcripts': [
                {
                    'asr_provider': 'Google STT Chirp',
                    'asr_transcript_text': 'alphabet_earnings_google.txt'
                },
                {
                    'asr_provider': 'AWS Transcribe',
                    'asr_transcript_text': 'alphabet_earnings_aws.txt'
                },
                {
                    'asr_provider': 'Microsoft CSS',
                    'asr_transcript_text': 'alphabet_earnings_msft.txt'
                },
                {
                    'asr_provider': 'OpenAI Whisper (small-en)',
                    'asr_transcript_text': 'alphabet_earnings_whisper_small_en.txt'
                },
                {
                    'asr_provider': 'Deepgram Nova',
                    'asr_transcript_text': 'alphabet_earnings_nova.txt'
                }
            ]
        },
        {
            'title': 'The Bee Movie (2007) dialogue',
            'gold_standard_transcript_text': 'bee_movie_gold_standard.txt',
            'gold_standard_transcript_link': 'https://gist.github.com/MattIPv4/045239bc27b16b2bcf7a3a9a4648c08a#file-bee-movie-script',
            'asr_transcripts': [
                {
                    'asr_provider': 'Google STT Chirp',
                    'asr_transcript_text': 'bee_movie_google.txt'
                },
                {
                    'asr_provider': 'AWS Transcribe',
                    'asr_transcript_text': 'bee_movie_aws.txt'
                },
                {
                    'asr_provider': 'Microsoft CSS',
                    'asr_transcript_text': 'bee_movie_msft.txt'
                },
                {
                    'asr_provider': 'OpenAI Whisper (large-v2)',
                    'asr_transcript_text': 'bee_movie_whisper_large-v2.txt'
                },
                {
                    'asr_provider': 'Deepgram Nova',
                    'asr_transcript_text': 'bee_movie_nova.txt'
                }
            ]
        }
    ]

    sidebar, main_body = st.columns([1,2])

    with sidebar:

        comparisons = [f"{audio['title']} - {asr_transcript['asr_provider']}" for audio in sample_transcripts for asr_transcript in audio['asr_transcripts']]
        st.selectbox('Choose sample transcripts to compare', (['Default short text'] + comparisons), index=0, key='example_transcripts', help='Select a gold-standard transcript and an ASR transcript to compare it to', on_change=fill_example_transcripts, kwargs={'default_gold_standard': default_gold_standard_text, 'default_asr': default_asr_text, 'sample_transcripts': sample_transcripts}, disabled=False, label_visibility="visible")
        if 'gold_standard_transcript_link' in st.session_state:
            st.write(f"[*Gold standard transcript link*]({st.session_state['gold_standard_transcript_link']})")
        with st.form('transcript-form'):
            # gold_standard_transcript = st.text_area('Paste the gold standard transcript here', value=open('sotu_2023_transcript_nyt.txt', 'r').read(), height=400)
            # asr_transcript = st.text_area('Paste the ASR transcript here', value=open('sotu_2023_transcript_whisper_small_en.txt', 'r').read(), height=400)
            gold_standard_transcript = st.text_area('Type or paste a gold standard transcript here', help='Paste a gold standard transcript here. A gold standard transcript should ideally have been generated manually by a human in order to guarantee accuracy and thoroughness.', key='gold_standard_transcript', height=400)
            asr_transcript = st.text_area('Type or paste an ASR transcript here', help='Paste an automatic speech recognition (ASR) transcript here. This transcript will be compared to the gold standard transcript in order to measure how accurate the ASR transcript was.', key='asr_transcript', height=400)
            with st.expander('View advanced settings', expanded=False):
                st.subheader('Advanced settings')
                st.write('**Text normalization options**')
                st.checkbox('Lower-case everything', value=True, key='normalize_lower_case', label_visibility="visible", help='Convert to lower-case text before comparing')
                st.checkbox('Standardize numbers', value=True, key='normalize_standardize_numbers', label_visibility="visible", help='Change numbers to words before comparing')
                st.checkbox('Expand contractions', value=True, key='normalize_expand_contractions', label_visibility="visible", help='Expand common contractions (e.g. \'weren\'t\' becomes \'were not\') before comparing')
                st.checkbox('Remove punctuation', value=True, key='normalize_remove_punctuation', label_visibility="visible", help='Remove punctuation (e.g. commas, periods, exclamation marks, etc.) before comparing')
                st.checkbox('Replace hyphens with spaces', value=True, key='normalize_replace_hyphens', label_visibility="visible", help='Replace hyphens and dashes with spaces before comparing')
                st.checkbox('Remove extra spaces', value=True, key='normalize_remove_spaces', label_visibility="visible", help='Remove extra spaces between words before comparing')
                st.write('**Minimum words options**')
                st.slider('Minimum words in similarity comparison', min_value=1, max_value=10, value=2, key='min_words_in_similarity_comparison', label_visibility="visible", help='Select the minimum number of words in either transcript to include when evaluating semantic similarity between *differing* sections (adjacent words will be incorporated if either of the two transcripts\' error sections is below the minimum)')
                st.caption("Error sections - words, phrases, or sentences where the two transcripts differ - may at times contain blank text in one of the transcripts. (For example, if one transcript states 'She walked to the office' and the second transcript states 'She walked to office', the error section for the second transcript would be blank, as the word 'the' is not present.) In this scenario, because an empty text string cannot be vector-embedded or compared to the embedding of another string, it is necessary to include surrounding words to ensure that the context of the differing sections is accounted for when determining semantic similarity.")
                st.write('**MinimumcCosine similarity options**')
                st.slider('Minimum cosine similarity threshold', min_value=0.0, max_value=1.0, value=0.70, step=0.01, key='minimum_similarity', label_visibility="visible", help='Select the cosine similarity threshold value below which an error section should be considered fully semantically different from the corresponding section in the gold standard transcript')
                st.caption("Although the technical range of cosine similarity is -1 to 1, in practice most word, phrase, and sentence pairs embedded by OpenAI's 'text-embedding-ada-002' model from corresponding transcripts have a cosine similarity between approximately 0.70 and 1.")

            evaluate_button = st.form_submit_button('Measure accuracy', type='primary')

        with main_body:
            body_container = st.container()
            col1, col2, col3, col4 = body_container.columns(4)
            with st.expander('**How It Works**', expanded=False):
                st.markdown(how_it_works(), unsafe_allow_html=True)
            loading_spinner_container = st.empty()
            worst_offenders_container = st.expander('**Worst Offenders**', expanded=True)
            transcript_container = st.expander('**Transcript Differences**', expanded=True)

        if evaluate_button or st.session_state['run_comparison_automatically']:
            with loading_spinner_container:
                with st.spinner('Normalizing the transcripts...'):
                    st.session_state['run_comparison_automatically'] = False
                    normalizations = []
                    for k in st.session_state:
                        if k.startswith('normalize_') and st.session_state[k] == True:
                            normalizations.append(k.replace('normalize_', ''))

                    normalized_text1 = normalize_text(gold_standard_transcript, normalizations=normalizations)
                    normalized_text2 = normalize_text(asr_transcript, normalizations=normalizations)

                with st.spinner('Calculating metrics...'):

                    if normalized_text1 == normalized_text2:
                        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
                        col1.metric(label="Total Words", value=f"{locale.format_string('%d', len(normalized_text1.split(' ')), grouping=True)}", help="Total words in the gold standard transcript")
                        col2.metric(label="WER", value=f"0%", help="Word Error Rate - (substitutions + insertions + deletions) / total words")
                        col3.metric(label="Distinct Error Sections", value="0", help="Number of distinct sections of the transcript containing one or more consecutive errors")
                        col4.metric(label="Semantic WER", value="0%", help="The semantic-adjusted Word Error Rate, in which the difference between the semantic meanings in each error section pair is multiplied by the number of error words in that section")
                        body_container.write('**The two normalized transcripts are identical.**')
                        st.stop()

                with st.spinner('Comparing the transcripts...'):

                    segments = compare_texts(normalized_text1, normalized_text2, model='jiwer', min_words=st.session_state['min_words_in_similarity_comparison'])
                    all_errors = [x for x in segments['grouped_segments'] if 'cosine_similarity' in x]

                    substitutions = sum([x['substitution_size'] * (1 - x['cosine_similarity']) for x in segments['grouped_segments'] if 'cosine_similarity' in x])
                    insertions = sum([x['insertion_size'] * (1 - x['cosine_similarity']) for x in segments['grouped_segments'] if 'cosine_similarity' in x])
                    deletions = sum([x['deletion_size'] * (1 - x['cosine_similarity']) for x in segments['grouped_segments'] if 'cosine_similarity' in x])
                    semantic_wer = (substitutions + insertions + deletions) / len(segments['alignments'].references[0])

                    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
                    col1.metric(label="Total Words", value=f"{locale.format_string('%d', len(segments['alignments'].references[0]), grouping=True)}", help="Total words in the gold standard transcript")
                    col2.metric(label="WER", value=f"{round(segments['alignments'].wer*100,2)}%", help="Word Error Rate - (substitutions + insertions + deletions) / total words")
                    col3.metric(label="Distinct Error Sections", value=f"{len([x for x in segments['grouped_segments'] if 'cosine_similarity' in x])}", help="Number of distinct sections of the transcript containing one or more consecutive errors")
                    col4.metric(label="Semantic WER", value=f"{round(semantic_wer*100, 2)}%", help="The semantic-adjusted Word Error Rate, in which the difference between the semantic meanings in each error section pair is multiplied by the number of error words in that section")
                    worst_offenders = [{'Gold Standard': f"{str(x['baseline_snippet_pre'] or '')} <strong style='color:red;'>{x['baseline_snippet']}</strong> {str(x['baseline_snippet_post'] or '')}", 'Error Section': f"{str(x['comparison_snippet_pre'] or '')} <strong style='color:red;'>{x['comparison_snippet']}</strong> {str(x['comparison_snippet_post'] or '')}", 'Similarity Score': x['cosine_similarity']} for x in sorted(all_errors, key=lambda x: x['cosine_similarity'])[:20]]
                    df = pd.DataFrame.from_dict(worst_offenders)
                    with worst_offenders_container:
                        st.markdown(df.style.format({'Similarity Score': '{:.3f}'}).to_html(),unsafe_allow_html=True) # See https://discuss.streamlit.io/t/unable-to-center-table-cell-values-with-pandas-style-need-input-to-see-if-this-is-even-possible-with-streamlit/31852/2
                    with transcript_container:
                        annotated_text(display_annotated_transcripts(segments))

demo_streamlit_app()