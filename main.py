import functions_framework
import pandas as pd
import json
import re
import openai
import nltk
from collections import Counter
from textblob import TextBlob
from transformers import pipeline
from flask import request, jsonify
from google.cloud import bigquery


# ✅ Download and Load Stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

# ✅ Load Sentiment Analysis Pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# ✅ Initialize BigQuery Client
bigquery_client = bigquery.Client()

# ✅ OpenAI API Key (Replace with a Secure Method)
openai.api_key = "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"

# ✅ Helper Functions (Unchanged)
def extract_number(text):
    match = re.search(r"(\d+(?:\.\d+)?)", str(text))
    return float(match.group(1)) if match else None

def remove_stopwords(text):
    if isinstance(text, str):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return " ".join(filtered_words)
    return text

def extract_hashtags(caption):
    return re.findall(r"#(\w+)", caption) if isinstance(caption, str) else []

def get_most_common_word(text):
    if isinstance(text, str):
        words = re.findall(r'\b\w+\b', text.lower())
        words = [word for word in words if word not in stop_words]
        word_counts = Counter(words)
        return word_counts.most_common(1)[0] if word_counts else (None, 0)
    return None, 0

def get_text_length(text):
    return len(text.split()) if isinstance(text, str) else 0

def count_items(item_list):
    if isinstance(item_list, list):
        return len(item_list)
    elif isinstance(item_list, str):
        return len(item_list.split(',')) if item_list else 0
    return 0

def analyze_text(text):
    if not isinstance(text, str) or not text.strip():
        return "neutral", 0.5, 0, 0.5

    sentiment_result = sentiment_pipeline(text)[0]
    sentiment, confidence = sentiment_result["label"], sentiment_result["score"]
    polarity, subjectivity = TextBlob(text).sentiment.polarity, TextBlob(text).sentiment.subjectivity

    return sentiment, confidence, polarity, subjectivity

def extract_insights(text):
    prompt = f"""
    Analyze the following text and extract the main theme, tone, and any calls to action.

    Text: "{text}"

    Provide the response in JSON format with the following keys:
    - "main_theme": One word or phrase summarizing the main topic
    - "tone": Overall tone (e.g., persuasive, informative, casual, formal)
    - "call_to_action": List any direct calls to action found in the text (e.g., 'follow me', 'subscribe', 'comment below')

    Return only valid JSON output.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert marketing text analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        gpt_summary = response.choices[0].message.content.strip()
        parsed_data = json.loads(re.sub(r'```(json)?', '', gpt_summary).strip())

        return {
            "main_theme": parsed_data.get("main_theme", ""),
            "tone": parsed_data.get("tone", ""),
            "call_to_action": parsed_data.get("call_to_action", [])
        }

    except Exception as e:
        return {"main_theme": "", "tone": "", "call_to_action": []}

@functions_framework.http
def transform_video_data_handler(request):
    """HTTP-triggered function to transform video data and insert into BigQuery."""
    try:
        request_json = request.get_json()
        if not request_json or "merged_df" not in request_json:
            return jsonify({"error": "Missing required field: merged_df"}), 400

        merged_df = pd.DataFrame(request_json["merged_df"])

        print("🔍 DataFrame Columns:", merged_df.columns.tolist())

        # ✅ Data Transformation
        for col in ["avg_shot_len", "video_len", "longest_shot"]:
            merged_df[col] = merged_df[col].apply(extract_number)

        merged_df["processed_caption"] = merged_df["raw_caption"].apply(remove_stopwords)
        merged_df["all_text"] = merged_df["processed_caption"].astype(str) + " " + merged_df["processed_speech"].astype(str)
        merged_df["created_time"] = pd.to_datetime(merged_df["created_time"])
        merged_df["weekday"] = merged_df["created_time"].dt.day_name()
        merged_df["time_of_day"] = merged_df["created_time"].dt.time
        merged_df["time_of_day"] = merged_df["time_of_day"].astype(str)
        merged_df["hashtags"] = merged_df["raw_caption"].apply(extract_hashtags)
        merged_df["most_common_word"], merged_df["common_word_count"] = zip(*merged_df["all_text"].apply(get_most_common_word))
        merged_df["speech_length"] = merged_df["raw_speech"].apply(get_text_length)
        merged_df["caption_length"] = merged_df["raw_caption"].apply(get_text_length)
        merged_df["hashtag_count"] = merged_df["hashtags"].apply(count_items)
        merged_df["hashtags"] = merged_df["hashtags"].astype(str)
        merged_df["logo_count"] = merged_df["logos"].apply(count_items)

        # ✅ NLP Analysis
        merged_df[["sentiment", "confidence", "polarity", "subjectivity"]] = merged_df["all_text"].apply(
            lambda x: pd.Series(analyze_text(x))
        )

        merged_df["speech_rate"] = merged_df["speech_length"] / merged_df["video_len"]
        merged_df["words_per_frame"] = merged_df["speech_length"] / merged_df["shot_count"]
        merged_df["theme_repetition"] = merged_df["common_word_count"] / merged_df["speech_length"]

        merged_df[["main_theme", "tone", "call_to_action"]] = merged_df["all_text"].apply(
            lambda x: pd.Series(extract_insights(x))
        )

        merged_df["call_to_action"] = merged_df["call_to_action"].astype(str)

        print("✅ Data Transformation Complete")

        # ✅ Insert Transformed Data into BigQuery
        table_id = "bizbuddydemo-v3.client.sp_analyzed_posts"
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            autodetect=True,
        )

        print(merged_df.dtypes)

        job = bigquery_client.load_table_from_dataframe(merged_df, table_id, job_config=job_config)
        job.result()  # Wait for the job to complete

        print("✅ Data Successfully Inserted into BigQuery")

        return jsonify({"message": "Data processed and inserted into BigQuery"}), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500
