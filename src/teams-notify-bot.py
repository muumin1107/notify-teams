# -*- coding: utf-8 -*-
"""
AWS Lambda function to monitor arXiv RSS feeds, analyze papers using Amazon Bedrock
based on a hierarchical topic structure (L0-L6), and post the top 3 relevant papers
to Microsoft Teams as a single Adaptive Card.
"""

import os
import json
import requests
import feedparser
import boto3
from datetime import datetime, timedelta, timezone

# --- Environment Variables ---
# Fetched once during container initialization.

# AWS Services Configuration
TEAMS_WEBHOOK_URL = os.environ.get('TEAMS_WEBHOOK_URL')
DYNAMODB_TABLE_NAME = os.environ.get('DYNAMODB_TABLE_NAME')
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID')

# RSS Feeds (comma-separated URLs)
RSS_FEED_URLS_ENV = os.environ.get('RSS_FEED_URLS', '') 

# Processing Configuration
MAX_ARTICLES_TO_PROCESS_ENV = os.environ.get('MAX_ARTICLES_TO_PROCESS', '20')
ABSTRACT_TRUNCATE_LENGTH = 1500 # Max length for abstracts to save tokens

# Hierarchical Topic Environments (L0-L6)
TOPICS_L0_DOMAIN_ENV = os.environ.get('TOPICS_L0_DOMAIN', '')
TOPICS_L1_APPROACH_ENV = os.environ.get('TOPICS_L1_APPROACH', '')
TOPICS_L2_TASK_ENV = os.environ.get('TOPICS_L2_TASK', '')
TOPICS_L3_MODALITY_ENV = os.environ.get('TOPICS_L3_MODALITY', '')
TOPICS_L4_APPLICATION_ENV = os.environ.get('TOPICS_L4_APPLICATION', '')
TOPICS_L5_ENVIRONMENT_ENV = os.environ.get('TOPICS_L5_ENVIRONMENT', '')
TOPICS_L6_CHALLENGE_ENV = os.environ.get('TOPICS_L6_CHALLENGE', '')

# --- AWS Service Clients (Global) ---
# Initialize clients outside the handler for container reuse
try:
    dynamodb = boto3.resource('dynamodb')
    dynamodb_table = dynamodb.Table(DYNAMODB_TABLE_NAME)
    bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1') 
except Exception as e:
    print(f"[ERROR] Failed to initialize AWS clients: {e}")
    dynamodb = None
    dynamodb_table = None
    bedrock_runtime = None

JST = timezone(timedelta(hours=+9), 'JST')

def lambda_handler(event, context):
    """
    Main Lambda handler.
    Orchestrates the process of fetching, analyzing, and posting papers based on
    the L0-L6 hierarchical topic structure.
    """
    print("Process start...")

    # 1. Load and Validate Configurations
    
    # Check if AWS clients initialized correctly
    if not all([dynamodb, dynamodb_table, bedrock_runtime]):
        print("[FATAL] AWS clients are not initialized. Check IAM permissions and env vars.")
        return {'statusCode': 500, 'body': 'AWS Client initialization failed.'}
        
    # Parse processing limit
    try:
        max_to_process = int(MAX_ARTICLES_TO_PROCESS_ENV)
    except ValueError:
        max_to_process = 20
    print(f"Total maximum articles to present to LLM: {max_to_process}")

    # Parse RSS feeds
    try:
        feed_urls_list = [url.strip() for url in RSS_FEED_URLS_ENV.split(',') if url.strip()]
        if not feed_urls_list:
            print("Environment variable RSS_FEED_URLS is not set or empty. Exiting.")
            return {'statusCode': 200, 'body': 'RSS_FEED_URLS not set'}
        
        num_feeds = len(feed_urls_list)
        limit_per_feed = max(1, max_to_process // num_feeds) # Ensure at least 1
        print(f"Found {num_feeds} feeds. Fetching max {limit_per_feed} unprocessed articles per feed.")
        
    except Exception as e:
        print(f"[ERROR] Failed to parse RSS_FEED_URLS: {e}")
        return {'statusCode': 500, 'body': 'Failed to parse RSS_FEED_URLS'}

    # Parse L0-L6 Hierarchical Topics
    try:
        def parse_topics(env_var):
            return [t.strip() for t in env_var.split(',') if t.strip()]

        topic_hierarchy = {
            "L0_Domain": parse_topics(TOPICS_L0_DOMAIN_ENV),
            "L1_Approach": parse_topics(TOPICS_L1_APPROACH_ENV),
            "L2_Task": parse_topics(TOPICS_L2_TASK_ENV),
            "L3_Modality": parse_topics(TOPICS_L3_MODALITY_ENV),
            "L4_Application": parse_topics(TOPICS_L4_APPLICATION_ENV),
            "L5_Environment": parse_topics(TOPICS_L5_ENVIRONMENT_ENV),
            "L6_Challenge": parse_topics(TOPICS_L6_CHALLENGE_ENV)
        }
        
        if not any(topic_hierarchy.values()):
             print("All topic environment variables (TOPICS_L0_... etc.) are empty. Exiting.")
             return {'statusCode': 200, 'body': 'All topic environments are empty.'}
        
        print("Successfully loaded L0-L6 hierarchical topics.")
        
    except Exception as e:
        print(f"[ERROR] Failed to parse Topic Hierarchy environments: {e}")
        return {'statusCode': 500, 'body': 'Failed to parse Topic Hierarchy'}

    # 2. Collect Unprocessed Papers
    papers_to_analyze, processed_urls_in_this_run = collect_unprocessed_papers(
        feed_urls_list, limit_per_feed
    )

    if not papers_to_analyze:
        print("No new unprocessed articles found across all feeds.")
        return {'statusCode': 200, 'body': 'No new articles to analyze'}

    print(f"Collected {len(papers_to_analyze)} unprocessed articles total. Sending to LLM for analysis.")

    # 3. Analyze Papers via Bedrock
    selected_papers = []
    try:
        analysis_result = analyze_papers_with_bedrock(papers_to_analyze, topic_hierarchy)
        selected_papers = analysis_result.get("selected_papers", [])
    except Exception as e:
        print(f"[ERROR] An error occurred during Bedrock call: {e}")
        # Continue to DB update to avoid reprocessing errors

    # 4. Post Summary to Teams
    if selected_papers:
        print(f"LLM selected {len(selected_papers)} relevant articles. Posting to Teams.")
        try:
            success = post_summary_to_teams(selected_papers)
            if not success:
                print("[ERROR] Failed to post summary to Teams.")
        except Exception as e:
            print(f"[ERROR] An error occurred during Teams posting: {e}")
    else:
        print("LLM did not select any relevant articles from the batch.")

    # 5. Mark All Analyzed Papers as Processed
    print(f"Marking {len(processed_urls_in_this_run)} articles as processed in DynamoDB.")
    for url in processed_urls_in_this_run:
        mark_article_as_processed(url)

    print("Process complete.")
    return {'statusCode': 200, 'body': f'Process successful. Analyzed {len(papers_to_analyze)} articles.'}

# --- Helper Functions ---

def collect_unprocessed_papers(feed_urls, limit_per_feed):
    """
    Iterates over RSS feeds, collects unprocessed papers up to the limit.
    """
    papers_to_analyze = []
    processed_urls = []

    for feed_url in feed_urls:
        print(f"Processing feed: {feed_url}")
        
        try:
            feed = feedparser.parse(feed_url)
            if feed.bozo:
                print(f"[WARN] Failed to parse feed: {feed_url}, Exception: {feed.bozo_exception}")
                continue
        except Exception as e:
            print(f"[WARN] feedparser library error: {e} (URL: {feed_url})")
            continue

        if not feed.entries:
            print(f"Feed has no entries: {feed_url}")
            continue

        fetched_from_this_feed = 0
        
        for entry in feed.entries:
            if fetched_from_this_feed >= limit_per_feed:
                break 

            article_url = entry.link 

            if is_article_processed(article_url):
                continue

            # --- Found unprocessed paper ---
            fetched_from_this_feed += 1
            processed_urls.append(article_url)
            
            article_title = entry.title
            abstract = entry.summary
            
            truncated_abstract = abstract[:ABSTRACT_TRUNCATE_LENGTH]
            if len(abstract) > ABSTRACT_TRUNCATE_LENGTH:
                truncated_abstract += "... (truncated)"
                
            papers_to_analyze.append({
                "url": article_url,
                "title": article_title,
                "abstract": truncated_abstract
            })
            
    return papers_to_analyze, processed_urls

def is_article_processed(url):
    """
    Checks DynamoDB to see if the article URL has already been processed.
    """
    try:
        response = dynamodb_table.get_item(Key={'article_url': url})
        return 'Item' in response
    except Exception as e:
        print(f"[ERROR] DynamoDB get_item error: {e}")
        return False # Fail safe: treat as unprocessed

def mark_article_as_processed(url):
    """
    Writes the processed article URL to DynamoDB with a 7-day TTL.
    """
    ttl_timestamp = int((datetime.now(JST) + timedelta(days=7)).timestamp())
    try:
        dynamodb_table.put_item(
            Item={
                'article_url': url,
                'processed_at': datetime.now(JST).isoformat(),
                'ttl': ttl_timestamp
            }
        )
    except Exception as e:
        print(f"[ERROR] DynamoDB put_item error: {e}")

def analyze_papers_with_bedrock(papers, topic_hierarchy):
    """
    Calls Bedrock (Claude 3) to analyze a batch of papers using the
    L0-L6 hierarchical topic structure and advanced evaluation criteria.
    """
    
    topics_str = json.dumps(topic_hierarchy, ensure_ascii=False, indent=2)
    papers_str = json.dumps(papers, ensure_ascii=False, indent=2)

    prompt = f"""
    あなたは、複数の技術分野に精通した高度な学術専門家です。
    あなたの任務は、提示された「論文リスト」を、定義された「トピック階層」に基づいて厳密に分析・評価し、最も価値のある論文を選定することです。

    ## トピック階層 (評価基準)
    {topics_str}

    * L0_Domain: 学術ドメイン（研究分野）
    * L1_Approach: 技術アプローチ（手法）
    * L2_Task: タスク（目的・課題）
    * L3_Modality: データモダリティ（入力情報）
    * L4_Application: 応用分野（ユースケース）
    * L5_Environment: 実装環境・制約条件
    * L6_Challenge: 学習上の課題（研究テーマ的視点）

    ## 論文リスト (分析対象)
    {papers_str}

    ## 指示 (厳守)
    「論文リスト」の全論文を評価し、以下の「評価基準」に基づいて、最も価値が高いと判断される論文を**最大3件**まで選定してください。

    ### 評価基準 (優先度順)

    1.  **最高評価 (High Priority: 課題解決・応用融合)**
        * 複数のレイヤーにまたがる、価値の高い貢献をしている論文。
        * **(パターンA: 課題解決型)** L6（課題）に対し、新しいL1（手法）やL2（タスク）を提案し、解決策を示している。
        * **(パターンB: 応用融合型)** L1（手法）やL2（タスク）を、L4（応用分野）やL5（実装環境）に適用し、実用的な成果や新しい知見を提供している。

    2.  **中評価 (Medium Priority: 中核技術の革新)**
        * L1（手法）またはL2（タスク）自体に、従来技術を大幅に超えるような画期的な（State-of-the-Art）提案、または新しい概念を提示している。

    3.  **選定対象外 (Low Priority)**
        * L0（ドメイン）の総説（Survey）論文。
        * L3（モダリティ）のデータセット紹介のみ。
        * L4（応用分野）のみに関連する、技術的新規性の低い事例報告。

    ### 選定プロセス
    * 「論文リスト」全体から、「最高評価」（パターンA, B）に合致する論文のみを選定対象とします。
    * 「中評価」および「選定対象外」の基準に合致する論文は、選定しないでください。
    * 「最高評価」に合致した論文の中から、重要度が高い順に最大3件を選定してください。

    ### 出力形式 (JSON)
    選定した論文についてのみ、以下の情報をJSON形式で出力してください。
    JSONオブジェクトのみを返し、前後にテキストを含めないでください。

    1.  `url`: 論文のURL (論文リストからそのままコピー)
    2.  `title`: 論文のタイトル (論文リストからそのままコピー)
    3.  `matched_topic`: **最も主要な貢献**と判断したトピック名（L1, L2, L4, L6などから）。**複数該当する場合は文字列のリスト**とすること。
    4.  `summary`: アブストラクトの核心的な内容を、日本語で200文字程度に簡潔に要約してください。
    5.  `keywords`: 論文の主要な日本語キーワードを**3件**程度のリスト。

    例 (1件選定された場合):
    {{
      "selected_papers": [
        {{
          "url": "http://arxiv.org/abs/...",
          "title": "農業用少数ショット学習",
          "matched_topic": ["少数ショット", "農業ロボット工学"],
          "summary": "少数ショット(L6)という課題に対し、新しい対照学習(L1)を提案。農業分野(L4)での作物分類(L2)タスクで有効性を実証。",
          "keywords": ["少数ショット", "対照学習", "農業ロボット工学"]
        }}
      ]
    }}
    
    例 (該当なしの場合):
    {{
      "selected_papers": []
    }}
    """

    body_json = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096, 
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    })
    
    print("Sending request to Bedrock...")
    try:
        response = bedrock_runtime.invoke_model(
            body=body_json, 
            modelId=BEDROCK_MODEL_ID, 
            contentType='application/json', 
            accept='application/json'
        )
        
        response_body = json.loads(response.get('body').read())
        result_text = response_body['content'][0]['text']
        
        print(f"Bedrock raw response (result_text): {result_text}")

        try:
            # Extract JSON part from potential text enclosures
            if '{' in result_text and '}' in result_text:
                json_part = result_text[result_text.find('{') : result_text.rfind('}')+1]
                analysis_data = json.loads(json_part)
            else:
                analysis_data = json.loads(result_text)

            print(f"Parsed JSON data: {analysis_data}")
            
            # Validate response structure
            if 'selected_papers' not in analysis_data or not isinstance(analysis_data['selected_papers'], list):
                print("[WARN] Bedrock response missing 'selected_papers' (list).")
                return {"selected_papers": []}
                
            return analysis_data
        
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON parse error: {e}. Response text: {result_text}")
            return {"selected_papers": []}

    except Exception as e:
        print(f"[ERROR] Bedrock API call error: {e}")
        # Re-raise the exception to be caught by the handler, 
        # so DB update still occurs but the error is logged.
        raise e

def post_summary_to_teams(selected_papers):
    """
    Posts a single Adaptive Card to Teams summarizing the selected papers,
    using the user-specified TextBlock Markdown layout.
    """
    
    card_body_elements = []

    for i, paper in enumerate(selected_papers):
        
        # Add a separator line between papers (starting from the second paper)
        if i > 0:
            card_body_elements.append({"type": "TextBlock", "text": "---", "separator": True})

        # 1. Paper Title (with ranking)
        card_body_elements.append({
            "type": "TextBlock",
            "text": f"**{i+1}. {paper.get('title', '(No Title)')}**",
            "size": "Medium",
            "weight": "Bolder",
            "wrap": True
        })
        
        # 2. Topic, Keywords, and Summary in a single Markdown TextBlock
        
        # ▼▼▼ MODIFICATION: Handle list or string for matched_topic ▼▼▼
        topic_value = paper.get('matched_topic', '(N/A)')
        if isinstance(topic_value, list):
            if not topic_value: # Handle empty list
                topic_str = '(N/A)'
            else:
                topic_str = ", ".join(topic_value) # Join list into string
        else:
            topic_str = str(topic_value) # Fallback for single string
        # ▲▲▲ MODIFICATION END ▲▲▲
            
        keywords_str = ", ".join(paper.get('keywords', []))
        summary_str = paper.get('summary', '(No Summary)')

        # \n\n creates a new paragraph (visual gap) in Markdown
        markdown_text = (
            f"**該当トピック:**\n{topic_str}\n\n"
            f"**キーワード:**\n{keywords_str}\n\n"
            f"{summary_str}"
        )

        card_body_elements.append({
            "type": "TextBlock",
            "text": markdown_text,
            "wrap": True,
            "spacing": "Medium" # Space from the title
        })

        # 3. Action button (Link to paper)
        card_body_elements.append({
            "type": "ActionSet",
            "actions": [
                {
                    "type": "Action.OpenUrl",
                    "title": "元の論文 (arXiv) を読む",
                    "url": paper.get('url', '#')
                }
            ],
            "spacing": "Medium" # Space from the summary block
        })

    # --- Full Adaptive Card Payload ---
    payload = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.5",
                    "msteams": {
                        "width": "Full" # Use full width
                    },
                    "body": card_body_elements
                }
            }
        ]
    }
    
    if not TEAMS_WEBHOOK_URL:
        print("[ERROR] TEAMS_WEBHOOK_URL is not set. Cannot post message.")
        return False
    
    try:
        response = requests.post(TEAMS_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        print(f"Teams post result: {response.status_code}")
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"[ERROR] Teams post error: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Teams error response: {e.response.text}")
        return False