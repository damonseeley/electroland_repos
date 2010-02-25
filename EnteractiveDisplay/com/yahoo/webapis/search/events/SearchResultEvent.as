/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.search.events
{
	import flash.events.Event;
	
	
	/**
	* Event class in response to result events from the Yahoo! Search API.
	* 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Alaric Cole 02/22/07
	*/
	public class SearchResultEvent extends Event
	{
		/** Constant for the event types. */
		
		public static const RESULT:String = "result";
		
		public static const WEB_SEARCH_RESULT:String = "webSearchResult";
		
		
		public static const NEWS_SEARCH_RESULT:String = 'newsSearchResult';
		
		public static const IMAGE_SEARCH_RESULT:String = 'imageSearchResult';
		public static const VIDEO_SEARCH_RESULT:String = 'videoSearchResult';
		
		public static const AUDIO_SEARCH_RESULT:String = 'audioSearchResult';
		public static const PODCAST_SEARCH_RESULT:String = 'podcastSearchResult';
		public static const ARTIST_SEARCH_RESULT:String = 'artistSearchResult';
		public static const ALBUM_SEARCH_RESULT:String = 'albumSearchResult';
		public static const SONG_SEARCH_RESULT:String = 'songSearchResult';
		public static const SONG_DOWNLOAD_LOCATION_SEARCH_RESULT:String = 'songDownloadLocationSearchResult';
		
		public static const LOCAL_SEARCH_RESULT:String = 'localSearchResult';
		
		public static const RELATED_SUGGESTION_RESULT:String = 'relatedSuggestionSearchResult';
		public static const SPELLING_SUGGESTION_RESULT:String = 'spellingSuggestionSearchResult';
		
		public static const TERM_EXTRACTION_RESULT:String = 'termExtractionResult';
		public static const CONTEXT_ANALYSIS_RESULT:String = 'contextAnalysisResult';
		
		
		private var _result:Object;
		
		private var _typeOfSearch:String;
		
		/**
		 * Constructs a new ResultEvent
		 */
		public function SearchResultEvent(type:String, inData:Object, typeOfSearch:String = null)
		{
			_result = inData;
			_typeOfSearch = typeOfSearch;
									   	
			super( type, bubbles, cancelable );
		}
		
		[Bindable]
		/**
		 * The result, usually an array of objects
		 */
		public function get result():Object
		{
			return _result;
		}
		
		public function set result( value:Object ):void
		{
			_result = value;
		}
		
		[Bindable]
		/**
		 * Passes the specific search result type (web search, image search, etc.)
		 */
		public function get typeOfSearch():String
		{
			return _typeOfSearch;
		}
		
		public function set typeOfSearch( value:String ):void
		{
			_typeOfSearch = value;
		}
	
	}
	
}