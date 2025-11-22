from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime
import os
from config import YOUTUBE_API_KEY

print("Connecting to YouTube...\n")

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

print("Searching for paracetamol videos with comments enabled...\n")
print("This will take a few minutes...\n")

all_comments = []
videos_checked = 0
videos_with_comments = 0
next_page_token = None

# We'll search in batches to get up to 1000 videos
for batch in range(10):  # 10 batches of 100 = 1000 videos
    print(f"Batch {batch + 1}/10...\n")
    
    search_request = youtube.search().list(
        q='paracetamol',
        part='snippet',
        type='video',
        maxResults=100,  # Max per batch
        order='relevance',
        pageToken=next_page_token
    )
    
    search_results = search_request.execute()
    next_page_token = search_results.get('nextPageToken')
    
    for item in search_results['items']:
        video_id = item['id']['videoId']
        video_title = item['snippet']['title']
        
        videos_checked += 1
        print(f"[{videos_checked}] Checking: {video_title[:50]} - ", end="")
        
        try:
            # Check if comments are enabled
            video_request = youtube.videos().list(
                id=video_id,
                part='statistics'
            )
            video_result = video_request.execute()
            
            if video_result['items']:
                stats = video_result['items'][0]['statistics']
                comment_count = int(stats.get('commentCount', 0))
                
                if comment_count > 0:
                    print(f"✓ {comment_count} comments")
                    videos_with_comments += 1
                    
                    # Get comments
                    comments_request = youtube.commentThreads().list(
                        videoId=video_id,
                        part='snippet',
                        textFormat='plainText',
                        maxResults=20,
                        order='relevance'
                    )
                    
                    comments_results = comments_request.execute()
                    
                    for comment_item in comments_results['items']:
                        comment = comment_item['snippet']['topLevelComment']['snippet']
                        
                        all_comments.append({
                            'text': comment['textDisplay'],
                            'author': comment['authorDisplayName'],
                            'likes': comment['likeCount'],
                            'video_title': video_title,
                            'video_id': video_id,
                            'published_at': comment['publishedAt'],
                            'source': 'youtube'
                        })
                else:
                    print("✗ No comments")
        
        except Exception as e:
            print(f"✗ Error")
    
    # Stop if no next page
    if not next_page_token:
        break
    
    print(f"\nTotal so far: {len(all_comments)} comments\n")

print(f"\n{'='*60}")
print(f"Videos checked: {videos_checked}")
print(f"Videos with comments: {videos_with_comments}")
print(f"Total comments found: {len(all_comments)}")
print(f"{'='*60}\n")

if all_comments:
    # Remove duplicates
    df = pd.DataFrame(all_comments)
    df = df.drop_duplicates(subset=['text'])
    
    # Create folder
    os.makedirs("data/raw", exist_ok=True)
    
    # Save to CSV
    filename = "data/raw/youtube_comments.csv"
    df.to_csv(filename, index=False)
    
    print(f"✓ Saved {len(df)} unique comments to {filename}")
    print("\nFirst 10 comments:")
    for i, text in enumerate(df['text'].head(10), 1):
        print(f"{i}. {text[:100]}...\n")
else:
    print("✗ No comments found.")
