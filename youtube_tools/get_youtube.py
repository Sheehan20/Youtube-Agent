#!/usr/bin/env python
# -*- coding: utf-8 -*-
# YouTube API Integration Module

import asyncio
import os
from typing import List, Dict
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def get_youtube_api():
    """Creates a YouTube API client."""
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY not found in environment variables")
    return build('youtube', 'v3', developerKey=api_key)

async def search_videos(keyword: str, max_results: int = 25) -> List[Dict]:
    """
    Searches for YouTube videos.

    Args:
        keyword: The search keyword.
        max_results: The maximum number of results to return, default is 25.

    Returns:
        List[Dict]: A list of video information dictionaries.
    """
    try:
        youtube = get_youtube_api()

        # Call YouTube Search API
        search_response = youtube.search().list(
            q=keyword,
            part='id,snippet',
            maxResults=max_results,
            type='video',
            order='relevance',  # Options: relevance, date, rating, viewCount
            regionCode='US',  # Can be changed to other regions
            relevanceLanguage='en'  # Can be changed to 'zh' for Chinese results
        ).execute()

        videos = []
        video_ids = []

        # Collect video IDs
        for item in search_response.get('items', []):
            if 'videoId' in item['id']:
                video_ids.append(item['id']['videoId'])

        # Batch fetch detailed statistics for videos
        stats_dict = {}
        if video_ids:
            stats_response = youtube.videos().list(
                part='statistics,contentDetails',
                id=','.join(video_ids)
            ).execute()

            # Create a dictionary for statistics
            for item in stats_response.get('items', []):
                stats_dict[item['id']] = item['statistics']

        # Assemble video information
        for item in search_response.get('items', []):
            if 'videoId' not in item['id']:
                continue

            video_id = item['id']['videoId']
            video_stats = stats_dict.get(video_id, {})

            video_info = {
                'video_id': video_id,
                'title': item['snippet']['title'],
                'channel': item['snippet']['channelTitle'],
                'description': item['snippet'].get('description', ''),
                'published_at': item['snippet']['publishedAt'],
                'thumbnail': item['snippet']['thumbnails']['default']['url'],
                'view_count': video_stats.get('viewCount', '0'),
                'like_count': video_stats.get('likeCount', '0'),
                'comment_count': video_stats.get('commentCount', '0'),
                'url': f'https://www.youtube.com/watch?v={video_id}'
            }
            videos.append(video_info)

        return videos

    except HttpError as e:
        print(f"YouTube API Error: {e}")
        return []
    except Exception as e:
        print(f"Error searching videos: {e}")
        return []

async def youtube_detail_pipeline(keywords: List[str], page: int = 1) -> List[Dict]:
    """
    Processes a list of keywords and returns formatted data.
    Similar to BiliBili's bilibili_detail_pipeline.

    Args:
        keywords: A list of keywords.
        page: Page number (YouTube API does not use page numbers, this parameter is kept for interface consistency).

    Returns:
        List[Dict]: A list of formatted video data.
    """
    all_results = []

    for keyword in keywords:
        print(f"Searching YouTube keyword: {keyword}")
        videos = await search_videos(keyword, max_results=25)

        # Format data to match existing structure
        formatted_data = []
        for video in videos:
            # Truncate description to avoid excessive length
            description = video['description'][:200] if video['description'] else "No description"

            video_text = (
                f"Type: video\n"
                f"Author: {video['channel']}\n"
                f"Video URL: {video['url']}\n"
                f"Title: {video['title']}\n"
                f"Description: {description}\n"
                f"Views: {video['view_count']}\n"
                f"Likes: {video['like_count']}\n"
                f"Comments: {video['comment_count']}\n"
                f"Published: {video['published_at']}\n"
            )
            formatted_data.append(video_text)

        all_results.append({
            "keyword": keyword,
            "real_data": str(formatted_data)
        })

    print(f"all_results: {all_results}")
    return all_results

if __name__ == '__main__':
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    async def main():
        results = await youtube_detail_pipeline(["Python tutorial"], page=1)
        print(f"Found {len(results)} results")

    asyncio.run(main())

