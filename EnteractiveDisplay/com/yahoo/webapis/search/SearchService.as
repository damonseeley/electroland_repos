/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.search
{
	import com.yahoo.webapis.IService;
	import com.yahoo.webapis.ServiceBase;
	import com.yahoo.webapis.search.events.SearchFaultEvent;
	import com.yahoo.webapis.search.events.SearchResultEvent;
	import com.yahoo.webapis.search.params.SearchParams;
	
	import flash.events.Event;
	import flash.events.EventDispatcher;
	import flash.events.HTTPStatusEvent;
	import flash.events.IOErrorEvent;
	import flash.events.ProgressEvent;
	import flash.events.SecurityErrorEvent;
	import flash.net.URLLoader;
	import flash.net.URLRequest;
	import flash.net.URLRequestMethod;
	import com.yahoo.webapis.ServiceFault;
	
	
	//--------------------------------------
	//  Events
	//--------------------------------------
	
	/**
	 *  Dispatched after a search is finished.
	 * 
	 *  @eventType com.yahoo.webapis.search.events.SearchResultEvent
	 */
	[Event(name="result", type="com.yahoo.webapis.search.events.SearchResultEvent")]

	/**
	 * Dispatched after a fault occurs.
	 * This may happen due to incorrect parameters being sent.
	 * @eventType com.yahoo.webapis.search.events.SearchFaultEvent
	 */
	[Event(name="fault", type="com.yahoo.webapis.search.events.SearchFaultEvent")]

	/**
	 * Yahoo! Search API Service Class.
	 * 
	 * <p>Yahoo! Search Web Services allow you to access Yahoo! content and services. </p>
	 *  
	 * 
	 * @mxml
	 *
	 *  <p>The <code>&lt;yahoo:SearchService&gt;</code> has the following tag attributes:</p>
	 *
	 *  <pre>
	 *  &lt;yahoo:SearchService
	 *    <b>Properties</b>
	 *    applicationId="YahooDemo"
	 *    type="web|image|audio|video|local|artist|album|song|songDownload|news|context|termExtraction|related|spellingSuggestion"
	 *    query="<i>No default</i>"
	 *    startAt="0"
	 *    maximumResults="10"
	 *    request="<i>No default</i>"
	 * 
	 *    <b>Events</b>
	 *    result="<i>No default</i>"
	 *    fault="<i>No default</i>"
	 *  /&gt;
	 *  </pre>
	 *
	 * <p>To use this class in MXML, do the following:</p>
	 *
	 *  <pre>
	 *    &lt;yahoo:SearchService applicationId="YahooDemo" type="web"/&gt;
	 *  </pre>
	 * 
	 * <p>Then give the tag a <code>query</code> attribute with your search terms and call the <code>send()</code> method.
	 * For more advanced usage, declare a <code>SearchParams</code> object, give it properties applicable to the type of search
	 * you are performing, and pass this to the <code>send()</code> method</p>
	 * 
	 * @see #type
	 * @see com.yahoo.webapis.search.params.SearchParams
	 * @see #send()
	 * 
	 * @example
	 * 
	 * <listing version="3.0" >
	 * &lt;?xml version="1.0" encoding="utf-8"?&gt;
	 * &lt;mx:Application 
	 * xmlns:mx="http://www.adobe.com/2006/mxml"  
	 * xmlns:yahoo="http://www.yahoo.com" &gt;
	 * 
	 * &lt;!-- The formatter to format the total results with commas --&gt;
	 * &lt;mx:NumberFormatter id="commaFormatter"/&gt;
	 * 
	 * &lt;!-- The Service for all types of searches --&gt;
	 * &lt;yahoo:SearchService id="searchService" applicationId="YahooDemo" query="{criteriaTextInput.text}" type="web" maximumResults="50"/&gt;
	 * 
 	 * &lt;!-- The form for searching --&gt;
	 *  &lt;mx:HBox defaultButton="{searchButton}"&gt;
	 *  	
	 * 	&lt;mx:TextInput id="criteriaTextInput"/&gt;
	 * 	&lt;mx:Button id="searchButton" label="search" click="searchService.send()"/&gt;
	 * 
	 * &lt;/mx:HBox&gt;
	 * 
	 * &lt;mx:Label text="{commaFormatter.format(searchService.numResultsAvailable)} results" visible="{Boolean(searchService.numResultsAvailable)}"/&gt;
	 *  
	 * &lt;!-- The results as a List --&gt;
	 * &lt;mx:List id="resultsList" showDataTips="true" dataProvider="{searchService.lastResult}" variableRowHeight="true" width="100%" height="100%"&gt;
	 * 	&lt;mx:itemRenderer&gt;
	 * 		&lt;mx:Component&gt;
	 * 			&lt;mx:VBox toolTip="{data.summary}" doubleClickEnabled="true" doubleClick="navigateToURL(new URLRequest(data.clickURL), '_blank')"&gt;
	 * 				&lt;mx:HBox &gt;
	 * 					&lt;mx:Label text="{data.index  + 1}" width="25"/&gt;
	 * 					&lt;mx:Text text="{data.name}" fontWeight="bold"/&gt;
	 * 				&lt;/mx:HBox&gt;
	 * 				&lt;mx:Image source="{data.thumbnail.url}" /&gt;
	 * 			 &lt;/mx:VBox&gt;
	 * 		&lt;/mx:Component&gt;
	 * 	&lt;/mx:itemRenderer&gt;
	 * &lt;/mx:List&gt;
	 * &lt;/mx:Application&gt;
 	 * </listing>
 	 * 
	 * @langversion ActionScript 3.0
	 * @playerversion Flash 9
	 * @author Alaric Cole 02/22/07
	 */
		
	public class SearchService extends ServiceBase implements IService
	{
		//--------------------------------------------------------------------------
	    //
	    //  Public Constants / Namespaces
	    //
	    //--------------------------------------------------------------------------
	 
	 	/**
		* @private
		*/
	 	protected static const NAMESPACE_SEARCH:Namespace = new Namespace("urn:yahoo:srch");
		/**
		* @private
		*/
	 	protected static const NAMESPACE_NEWS:Namespace = new Namespace("urn:yahoo:yn");
		/**
		* @private
		*/
	 	protected static const NAMESPACE_IMAGE:Namespace = new Namespace("urn:yahoo:srchmi");
		/**
		* @private
		*/
	 	protected static const NAMESPACE_AUDIO:Namespace = new Namespace("urn:yahoo:srchma");
		/**
		* @private
		*/
	 	protected static const NAMESPACE_AUDIO_ALT:Namespace = new Namespace("urn:yahoo:srchmm");
		/**
		* @private
		*/
	 	protected static const NAMESPACE_VIDEO:Namespace = new Namespace("urn:yahoo:srchmv");
		/**
		* @private
		*/
	 	protected static const NAMESPACE_TERM_EXTRACTION:Namespace = new Namespace("urn:yahoo:cate");
		/**
		* @private
		*/
	 	protected static const NAMESPACE_LOCAL:Namespace = new Namespace("urn:yahoo:lcl");
 		
	 	//--------------------------------------------------------------------------
	    //
	    //  Search Type Constants
	    //
	    //--------------------------------------------------------------------------
	    
	    /**
		* Yahoo! Web Search Web Services let you tap into the power of Yahoo! Search technologies 
		* from within other sites, applications, and environments. 
		*/
		public static const WEB_SEARCH:String = "web";
		
		/**
		* The News Search service allows you to search the Internet for news stories.
		*/
		public static const NEWS_SEARCH:String = "news";
		
		/**
		* The Image Search Web Service allows you to search the Internet for images.
		*/
		public static const IMAGE_SEARCH:String = "image";
		
		/**
		* The Video Search service allows you to search the Internet for video clips.
		*/
		public static const VIDEO_SEARCH:String = "video";
		
		/**
		* Yahoo! Audio Search Web Services provide access to music data and audio files using structured and unstructured queries.
		*/
		public static const AUDIO_SEARCH:String = "audio";
		
		/**
		* The Podcast Search service allows you to search the Internet for audio podcasts.
		*/
		public static const PODCAST_SEARCH:String = "podcast";
		
		/**
		* The Artist Search service allows you to find information on a particular musical performer. 
		*/
		public static const ARTIST_SEARCH:String = "artist";
		
		/**
		* The Album Search service allows you to find information on music albums.
		*/
		public static const ALBUM_SEARCH:String = "album";
		
		/**
		* The Song Search service allows you to find information on individual songs.
		*/
		public static const SONG_SEARCH:String = "song";
		
		/**
		* The Song Download Location service allows you to find places to download individual songs, from the web, and from various online music sources.
		*/
		public static const SONG_DOWNLOAD_LOCATION_SEARCH:String = "songDownload";
		
		/**
		* Yahoo! Local Search APIs offer you a way to connect with the wealth of data in Yahoo! Local — including ratings and comments made by Yahoo! users. 
		 * The content in Yahoo! Local makes a great addition to any mashup, bringing in location-based relevancy and the additional context of what real people have experienced in these places. 
		*/
		public static const LOCAL_SEARCH:String = "local";
		
		/**
		* The Related Suggestion service returns suggested queries to extend the power of a submitted query, providing variations on a theme to help you dig deeper.
		*/
		public static const RELATED_SUGGESTION:String = "related";
		
		/**
		* The Spelling Suggestion service provides a suggested spelling correction for a given term. 
		*/
		public static const SPELLING_SUGGESTION:String = "spellingSuggestion";
		
		/**
		* The Term Extraction Web Service provides a list of significant words or phrases extracted from a larger content. It is one of the technologies used in Y!Q.
		*/
		public static const TERM_EXTRACTION:String = "termExtraction";
		
		/**
		* The Contextual Web Search service allows you to search the Internet for web pages using a context-based query.
		*/
		public static const CONTEXT_ANALYSIS:String = "context";
		
		
	 	
		//--------------------------------------------------------------------------
	    //
	    //  Private variables
	    //
	    //--------------------------------------------------------------------------
	
		private var _numResultsAvailable:uint;
		private var _numResultsReturned:uint;
		private var _numTotalResultsReturned:uint;
		private var _firstResultPosition:uint;
		private var _lastResultPosition:uint;
		private var _autoIncrement:Boolean;
		
		private var _lastResult:Object = [];
		
		/**
		* The last search parameters sent
		* @private
		*/
		private var previousRequest:String;
		
		/**
		* The last type invoked
		* @private
		*/
		private var previousType:String = "";
		
		private var _maximumResults:uint = 10;
		
		//--------------------------------------------------------------------------
	    //
	    //  Search URLs
	    //
	    //--------------------------------------------------------------------------
		
		/**
		* 
		* @private
		*/
		protected static const BASE_URL:String = "http://search.yahooapis.com/";
 		
 		/**
		* 
		* @private
		*/
		protected static const WEB_SEARCH_URL:String = BASE_URL + "WebSearchService/V1/webSearch";
		/**
		* 
		* @private
		*/
		protected static const RELATED_SUGGESTION_URL:String = BASE_URL + "WebSearchService/V1/relatedSuggestion";
		/**
		* 
		* @private
		*/
		protected static const SPELLING_SUGGESTION_URL:String = BASE_URL + "WebSearchService/V1/spellingSuggestion";
		/**
		* 
		* @private
		*/
		protected static const CONTEXT_ANALYSIS_URL:String = BASE_URL + "WebSearchService/V1/contextSearch";

		/**
		* 
		* @private
		*/
		protected static const NEWS_SEARCH_URL:String = BASE_URL + "NewsSearchService/V1/newsSearch";
		
		/**
		* 
		* @private
		*/
		protected static const IMAGE_SEARCH_URL:String = BASE_URL + "ImageSearchService/V1/imageSearch";
		/**
		* 
		* @private
		*/
		protected static const VIDEO_SEARCH_URL:String = BASE_URL + "VideoSearchService/V1/videoSearch";
		
		/**
		* 
		* @private
		*/
		protected static const AUDIO_SEARCH_URL:String = BASE_URL + "AudioSearchService/V1/audioSearch";
		/**
		* 
		* @private
		*/
		protected static const PODCAST_SEARCH_URL:String = BASE_URL + "AudioSearchService/V1/podcastSearch";
		/**
		* 
		* @private
		*/
		protected static const ARTIST_SEARCH_URL:String = BASE_URL + "AudioSearchService/V1/artistSearch";
		/**
		* 
		* @private
		*/
		protected static const ALBUM_SEARCH_URL:String = BASE_URL + "AudioSearchService/V1/albumSearch";
		/**
		* 
		* @private
		*/
		protected static const SONG_SEARCH_URL:String = BASE_URL + "AudioSearchService/V1/songSearch";
		/**
		* 
		* @private
		*/
		protected static const SONG_DOWNLOAD_LOCATION_SEARCH_URL:String = BASE_URL + "AudioSearchService/V1/songDownloadLocation";
		
		/**
		* 
		* @private
		*/
		protected static const TERM_EXTRACTION_URL:String = BASE_URL + "ContentAnalysisService/V1/termExtraction";
		
		/**
		* 
		* @private
		*/
		protected static const LOCAL_SEARCH_URL:String = "http://local.yahooapis.com/LocalSearchService/V3/localSearch";
		
		//--------------------------------------------------------------------------
	    //
	    //  Constructor
	    //
	    //--------------------------------------------------------------------------
	
		/**
		* Constructor.
		*/
 		public function SearchService()
 		{
 			
 		}
		
		
		//--------------------------------------------------------------------------
	    //
	    //  Public variables
	    //
	    //--------------------------------------------------------------------------
	
		[Bindable] 
		[Inspectable(enumeration="web,image,audio,video,local,artist,album,song,songDownload,news,context,termExtraction,related,spellingSuggestion")]
		/**
		* The type of search to perform
		* @default SearchService.WEB_SEARCH
		*/
		//TODO: Consider making this an array, allowing multiple types of searches in parallel
		public var type:String = WEB_SEARCH;
		
		/**
		* The query string to search for, in lieu of a request object
		* @see #request
		*/
		public var query:String;
		
		/**
		* The request object
		* @see com.yahoo.webapis.search.params.SearchParams
		*/
		public var request:SearchParams = new SearchParams();
		
		/**
		* The result index to begin returning results at EXAMPLE: 50 would start returning results at 50
		* @default 0
		*/
		public var startAt:uint = 0;
		
		[Bindable]
		/**
		* The number of times this search has been performed; the "page" of results
		* @default 0
		*/
		public var increment:uint = 0;
		
		[Bindable]
		/**
		* If the search has more results pending
		* 
		*/
		public var hasResultsPending:Boolean = true;
		
		
		[Bindable]
		/**
		* The maximum number of results to return
		* @default 10
		* maximum value is 100
		*/
		public function get maximumResults():uint
		{
			return _maximumResults;
		}
		
		public function set maximumResults( value:uint ):void
		{
			//maximum that the Yahoo! API will allow is 100
			//TODO: others may be less, this maybe should be lower
			if (value > 100) value = 100;
			_maximumResults = value;
		}

	
		[Bindable]
		/**
		 * The total amount of results available 
		 */
		public function get numResultsAvailable():uint
		{
			return _numResultsAvailable;
		}
		
		public function set numResultsAvailable( value:uint ):void
		{
			_numResultsAvailable = value;
		}
		
		
		[Bindable]
		/**
		 * The position of the first result .
		 */
		public function get firstResultPosition():uint
		{
			return _firstResultPosition;
		}
		
		public function set firstResultPosition( value:uint ):void
		{
			_firstResultPosition = value;
		}
		
		[Bindable]
		/**
		 * The position of the last result .
		 */
		public function get lastResultPosition():uint
		{
			return _lastResultPosition;
		}
		
		public function set lastResultPosition( value:uint ):void
		{
			_lastResultPosition = value;
		}
		
		[Bindable]
		/**
		 * The amount of results of this page.
		 */
		public function get numResultsReturned():uint
		{
			return _numResultsReturned;
		}
		
		public function set numResultsReturned( value:uint ):void
		{
			_numResultsReturned = value;
		}
		
		[Bindable]
		/**
		 * The total amount of results returned.
		 */
		public function get numTotalResultsReturned():uint
		{
			return _numTotalResultsReturned;
		}
		
		public function set numTotalResultsReturned( value:uint ):void
		{
			_numTotalResultsReturned = value;
		}
		
		[Bindable]
		/**
		 * Whether to retrieve more results on the same search.
		 */
		public function get autoIncrement():Boolean
		{
			return _autoIncrement;
		}
		
		public function set autoIncrement( value:Boolean ):void
		{
			_autoIncrement = value;
		}
		
		[Bindable]
		/**
		* The result of the last service invocation
		* 
		*/
		public function get lastResult():Object
		{
			return _lastResult;
		}
		//NOTE: these should probably be private or protected.
		//This usage is due to a compiler bug, which gives "Ambiguous reference to ..." errors
		//cf. http://www.adobe.com/cfusion/knowledgebase/index.cfm?id=4a146409
		public function set lastResult( value:Object ):void
		{
			_lastResult = value;
		}
		
	
		//--------------------------------------------------------------------------
	    //
	    //  Public Methods
	    //
	    //--------------------------------------------------------------------------
	
		/**
		* Search based on type.
		*
		* This provides a way to use a standard method for search.
		* <p>Note: Invoking the <code>send()</code> method multiple times with the same parameters will retrieve more results for that search.</p>
		* @param parameters A String or SearchParams object to specify parameters
		* @see com.yahoo.webapis.search.params.SearchParams 
		*/
		public function send(parameters:Object = null):void
		{
			//allow query that is set at the level of this service; useful for databinding
			if(parameters == null)
			{
				if(request == null)
				{
					request = new SearchParams();
					
					
				}
				
				if(query != null) 
				{
					request.query = query;
				}
					
			}
			
			
			if(parameters is String)
			{
				request.query = parameters as String;
			}
		
			if(parameters is SearchParams)
			{
				request = parameters as SearchParams;
			} 
			
			
			//allow parameters that are set at the level of this service
			if(request.maximumResults==0) 
			{
				request.maximumResults = _maximumResults;
			
			} 	
			var newRequest:String = request.toString();
			
			//Take out the results and start parameters for comparison
			newRequest = newRequest.replace(/&start=\d+/,"");
			newRequest = newRequest.replace(/&results=\d+/,"");
			//trace(previousRequest, request)
			
			if((previousType == type) && (previousRequest == newRequest) && autoIncrement)
			{
				//continue to retrieve results for this same search
				increment ++;
			}
			else
			{
				//new query, so reset
				increment = 0;
				hasResultsPending = true;
				lastResult = [];
				
				//store the last query
				previousRequest = request.toString();
				previousRequest = previousRequest.replace(/&start=\d+/,"");
				previousRequest = previousRequest.replace(/&results=\d+/,"");
			}
			
			//continue retrieving results for this same query, if there are any more
			if(!hasResultsPending) return;
			
			//hold the last type
			previousType = type;
			
			
			//Set the index to start at, based on paging (Yahoo! APIs are 1-based)
			request.start = startAt + ( increment * request.maximumResults  ) + 1;
			
			switch(type)
			{
				case TERM_EXTRACTION: 
					searchTermExtraction( request );
					break;
				
				case CONTEXT_ANALYSIS: 
					searchContextAnalysis( request );
					break;
				
				case RELATED_SUGGESTION:
					searchRelatedSuggestions( request );
					break;
				
				case SPELLING_SUGGESTION:
					searchSpellingSuggestions( request );
					break;
				
				case VIDEO_SEARCH:
					searchVideos( request );
					break;
				
				case NEWS_SEARCH:
					searchNews( request );
					break;
				
				case IMAGE_SEARCH:
					searchImages(request );
					break;
				
				case AUDIO_SEARCH:
					searchAudio(request );
					break;
				
				case PODCAST_SEARCH:
					searchPodcasts( request );
					break;
				
				case ARTIST_SEARCH:
					searchArtists( request );
					break;
				
				case ALBUM_SEARCH:
					searchAlbums( request );
					break;
				
				case SONG_SEARCH:
					searchSongs( request );
					break;
				
				case SONG_DOWNLOAD_LOCATION_SEARCH:
					searchSongDownloads( request );
					break;
				
				case LOCAL_SEARCH:
					searchLocal( request );
					break;
				
				default:
					//default to Web Search
					searchWeb( request );
	
			}
		}
		
		
		/**
		* Search pages based on a query.
		* @param params SearchParams passed for using search filters.
		*/
		public function searchWeb(params:SearchParams = null):void
		{
			handleQueryLoading( WEB_SEARCH_URL,  SearchResultEvent.WEB_SEARCH_RESULT, params);
		}
		
		/**
		* Search local based on a query and location.
		* @param params SearchParams passed for using search filters.
		*/
		public function searchLocal(params:SearchParams = null):void
		{
			handleQueryLoading( LOCAL_SEARCH_URL,  SearchResultEvent.LOCAL_SEARCH_RESULT, params);
		}
		
		
		/**
		* Search images based on a query.
		* @param params SearchParams passed for using search filters.
		*/
		public function searchImages(params:SearchParams = null):void
		{
			handleQueryLoading( IMAGE_SEARCH_URL,  SearchResultEvent.IMAGE_SEARCH_RESULT, params);
		}
		
		
		/**
		* Search news based on a query.
		* @param params SearchParams passed for using search filters.
		*/
		public function searchNews(params:SearchParams = null):void
		{
			handleQueryLoading( NEWS_SEARCH_URL,  SearchResultEvent.NEWS_SEARCH_RESULT, params);
		}
		
		
		/**
		* Search audio based on a query.
		* @param params SearchParams passed for using search filters.
		*/
		public function searchAudio(params:SearchParams = null):void
		{
			handleQueryLoading( AUDIO_SEARCH_URL,  SearchResultEvent.AUDIO_SEARCH_RESULT, params);
		}
		
		/**
		* Search podcasts based on a query.
		* @param params SearchParams passed for using search filters.
		*/
		public function searchPodcasts(params:SearchParams = null):void
		{
			handleQueryLoading( PODCAST_SEARCH_URL,  SearchResultEvent.PODCAST_SEARCH_RESULT, params);
		}
		
		/**
		* Search albums based on an album, album ID, artist, or artist ID.
		* @param params SearchParams passed for using search filters.
		*/
		public function searchAlbums(params:SearchParams = null):void
		{
			handleQueryLoading( ALBUM_SEARCH_URL,  SearchResultEvent.ALBUM_SEARCH_RESULT, params);
		}
		
		/**
		* Search artists based on an artist or artist ID.
		* @param params SearchParams passed for using search filters.
		*/
		public function searchArtists(params:SearchParams = null):void
		{
			handleQueryLoading( ARTIST_SEARCH_URL,  SearchResultEvent.ARTIST_SEARCH_RESULT, params);
		}
		
		/**
		* Search songs based on an song, song ID, album, album ID, artist, or artist ID.
		* @param params SearchParams passed for using search filters.
		*/
		public function searchSongs(params:SearchParams = null):void
		{
			handleQueryLoading( SONG_SEARCH_URL,  SearchResultEvent.SONG_SEARCH_RESULT, params);
		}
		
		/**
		* Search for places to download songs based on a song ID
		* @param params SearchParams passed for using search filters.
		*/
		public function searchSongDownloads(params:SearchParams = null):void
		{
			handleQueryLoading( SONG_DOWNLOAD_LOCATION_SEARCH_URL,  SearchResultEvent.SONG_DOWNLOAD_LOCATION_SEARCH_RESULT, params);
		}
		
		
		/**
		* Search video based on a query.
		* @param params SearchParams passed for using search filters.
		*/
		public function searchVideos(params:SearchParams = null):void
		{
			handleQueryLoading( VIDEO_SEARCH_URL,  SearchResultEvent.VIDEO_SEARCH_RESULT, params);
		}
		
		/**
		* Search spelling suggestions based on a misspelled word.
		* @param params SearchParams passed for using search filters.
		*/
		public function searchSpellingSuggestions(params:SearchParams = null):void
		{
			handleQueryLoading( SPELLING_SUGGESTION_URL,  SearchResultEvent.SPELLING_SUGGESTION_RESULT, params);
		}
		
		/**
		* Search related suggestions based on a query.
		* @param params SearchParams passed for using search filters.
		*/
		public function searchRelatedSuggestions(params:SearchParams = null):void
		{
			handleQueryLoading( RELATED_SUGGESTION_URL,  SearchResultEvent.RELATED_SUGGESTION_RESULT, params);
		}
		
		/**
		* Search keywords based on context.
		* @param params SearchParams passed for using search filters.
		*/
		public function searchTermExtraction(params:SearchParams = null):void
		{
			handleQueryLoading( TERM_EXTRACTION_URL,  SearchResultEvent.TERM_EXTRACTION_RESULT, params);
		}
		
		/**
		* Search pages based on a query and context.
		* @param params SearchParams passed for using search filters.
		*/
		public function searchContextAnalysis(params:SearchParams = null):void
		{
			handleQueryLoading( CONTEXT_ANALYSIS_URL,  SearchResultEvent.CONTEXT_ANALYSIS_RESULT, params);
		}
		
		
		
		//--------------------------------------------------------------------------
	    //
	    //  Private Methods
	    //
	    //--------------------------------------------------------------------------		
		
		/**
		* Internal handling and loading a Search API Call.
		* @private
		* @param inSendMethod 	The String description of the Method.
		* @param dispatchType The Event dispatch type.
		* @param params 		Value for passing Param object.
		* @param inDepth 		An Array containing: 0 = Search or Caches, 1 = XML Child Depth 1, 2 = XML Child Depth 2.
		*/
		protected function handleQueryLoading(inSendMethod:String, dispatchType:String, params:SearchParams = null, inDepth:Array = null):void
		{
			
			var sendQuery:String = (inSendMethod + "?appid=" + applicationId );
			var typeQuery:String;
			
			if(params != null)
			{
				var paramString:String = params.collect();
				sendQuery += ("&"+paramString);
			}
			
			
 			var queryXMLURLRequest:URLRequest = new URLRequest(sendQuery);
 			
 			//make the query XML in POST format for potentially long requests
 			queryXMLURLRequest.method = URLRequestMethod.POST;
 			
			var queryLoader:URLLoader = new URLLoader(queryXMLURLRequest);
			queryLoader.addEventListener(Event.COMPLETE, completeHandler);
			queryLoader.addEventListener(IOErrorEvent.IO_ERROR, ioErrorHandler);
			queryLoader.addEventListener(Event.OPEN, openHandler);
            queryLoader.addEventListener(ProgressEvent.PROGRESS, progressHandler);
            queryLoader.addEventListener(SecurityErrorEvent.SECURITY_ERROR, securityErrorHandler);
            queryLoader.addEventListener(HTTPStatusEvent.HTTP_STATUS, httpStatusHandler);
           	
         // XML loading result method
		/**
		* @private
		*/
		function completeHandler(event:Event):void
			{
				//trace("completeHandler: " + event);

				var queryXML:XML = new XML(event.currentTarget.data);
				
				if(queryXML.Error == undefined)
				{
					// Determine if it's a (Search or Cache List) or a WebSearchResult List
					if( hasOwnProperty("inDepth") )
					{
						if(inDepth.length > 0)
						{
							
							var resultsArray:Array = [];
							
							if(resultsArray.length > 0)
							{
								dispatchResult( resultsArray, dispatchType);
							}
	
						}
						
					}
					else
					{
						// Format and Dispatch the Normal WebSearchResult Objects
						//queryXML = new XML(queryLoader.data);
						formatAndDispatch(queryXML, dispatchType);
					}
					
				}
				else
				{
					dispatchFault(new ServiceFault(SearchFaultEvent.API_RESPONSE, queryXML.Error, queryXML.Error.Message ));
				}
			    
			}
		}
		
			
			
		/**
		* @private
		*/
		private function ioErrorHandler(event:IOErrorEvent):void
		{
		    dispatchFault(new ServiceFault(SearchFaultEvent.XML_LOADING,"The URL could not be found", event.text ));
		}
			
        /**
		* @private
		*/
		private function openHandler(event:Event):void 
		{
            //trace("openHandler: " + event);
        }

        /**
		* @private
		*/
		private function progressHandler(event:ProgressEvent):void 
		{
            //it is not possible to access the data until it has been received completely
            //so bytes total = bytes loaded
            //trace("progressHandler loaded:" + event.bytesLoaded + " total: " + event.bytesTotal);
        }

        /**
		* @private
		*/
		private function securityErrorHandler(event:SecurityErrorEvent):void 
		{
            //trace("securityErrorHandler: " + event);
        }

		/**
		* @private
		*/
        private function httpStatusHandler(event:HTTPStatusEvent):void 
		{
            //trace("httpStatusHandler: " + event);
        }

              
		/**
		* @private
		* Recurse lastResult and transcribe into Search Objects, then dispatch result Event.
		* @param inXML			 The XML to transcribe into Objects.
		* @param dispatchType The term of the dispatched Event.
		*/
		protected function formatAndDispatch(inXML:XML, dispatchType:String):void
		{
		    if(increment == 0) lastResult = [];
		    
		    var len:int = lastResult.length;
		    
		    numResultsAvailable = inXML.@totalResultsAvailable;
		    numResultsReturned = inXML.@totalResultsReturned;
		    numTotalResultsReturned = numResultsReturned + lastResult.length;
		    firstResultPosition = inXML.@firstResultPosition;
		    lastResultPosition = firstResultPosition + len;
		    
		    //moreSearchLink = inXML.@moreSearch;
		    
			var resultsArray:Array = [];
		    
		    var resultNode:XML;
		    
		    
		    //loop through child nodes and build value objects based on search type
		    switch(dispatchType)
			{
				case SearchResultEvent.LOCAL_SEARCH_RESULT:
					for each (resultNode in inXML.children())
						{
							var localSearchResult:LocalSearchResult = new LocalSearchResult();
							
							// LocalearchResult Properties
							//index is the current index in this set of results added to what has already been returned, if anything
							localSearchResult.index = resultNode.childIndex() + len;
							localSearchResult.name = resultNode.NAMESPACE_LOCAL::Title;
							localSearchResult.id = resultNode.@id;
							
							localSearchResult.url = resultNode.NAMESPACE_LOCAL::Url;
							localSearchResult.clickURL = resultNode.NAMESPACE_LOCAL::ClickUrl;
							
							localSearchResult.mapURL = resultNode.NAMESPACE_LOCAL::MapUrl;
							localSearchResult.businessClickURL = resultNode.NAMESPACE_LOCAL::BusinessClickUrl;
							localSearchResult.businessURL = resultNode.NAMESPACE_LOCAL::BusinessUrl;
							
							localSearchResult.address = resultNode.NAMESPACE_LOCAL::Address;
							localSearchResult.city = resultNode.NAMESPACE_LOCAL::City;
							localSearchResult.state = resultNode.NAMESPACE_LOCAL::State;
							localSearchResult.phone = resultNode.NAMESPACE_LOCAL::Phone;
							
							localSearchResult.latitude = resultNode.NAMESPACE_LOCAL::Latitude;
							localSearchResult.longitude = resultNode.NAMESPACE_LOCAL::Longitude;
							localSearchResult.distance = resultNode.NAMESPACE_LOCAL::Distance;
							
							localSearchResult.rating = new Rating();
							localSearchResult.rating.averageRating = resultNode.NAMESPACE_LOCAL::Rating.NAMESPACE_LOCAL::AverageRating;
							localSearchResult.rating.numRatings = resultNode.NAMESPACE_LOCAL::Rating.NAMESPACE_LOCAL::TotalRatings;
							localSearchResult.rating.numReviews = resultNode.NAMESPACE_LOCAL::Rating.NAMESPACE_LOCAL::TotalReviews;
							localSearchResult.rating.summary = resultNode.NAMESPACE_LOCAL::Rating.NAMESPACE_LOCAL::LastReviewIntro;
							//convert string to number to date
							localSearchResult.rating.lastReviewDate = new Date(resultNode.NAMESPACE_LOCAL::Rating.NAMESPACE_LOCAL::LastReviewDate as Number);
							
							localSearchResult.categories = [];
							
							for each(var l:XML in resultNode.NAMESPACE_LOCAL::Categories.NAMESPACE_LOCAL::Category)
							{
								var category:Category = new Category();
								category.name = l.toString();
								category.id = l.@id;
								
								localSearchResult.categories.push(category);
							}
							
							// Add the SearchResult Object to our Data Array.
							resultsArray.push(localSearchResult);
							
						}
					
				break;
				
				case SearchResultEvent.TERM_EXTRACTION_RESULT:
					for each (resultNode in inXML.children())
						{
							var termExtractionSearchResult:SearchResult = new SearchResult();
							
							// WebSearchResult Properties
							//index is the current index in this set of results added to what has already been returned, if anything
							termExtractionSearchResult.index = resultNode.childIndex() + len;
							termExtractionSearchResult.name = resultNode.toString();
							
							// Add the WebSearchResult Object to our Data Array.
							resultsArray.push(termExtractionSearchResult);
							
						}
					
				break;
				
				
				case SearchResultEvent.CONTEXT_ANALYSIS_RESULT:
					for each (resultNode in inXML.children())
						{
							var contextAnalysisSearchResult:WebSearchResult = new WebSearchResult();
							
							// WebSearchResult Properties
							//index is the current index in this set of results added to what has already been returned, if anything
							contextAnalysisSearchResult.index = resultNode.childIndex() + len;
							contextAnalysisSearchResult.name = resultNode.NAMESPACE_SEARCH::Title;
							contextAnalysisSearchResult.summary = resultNode.NAMESPACE_SEARCH::Summary;
							
							contextAnalysisSearchResult.url = resultNode.NAMESPACE_SEARCH::Url;
							contextAnalysisSearchResult.clickURL = resultNode.NAMESPACE_SEARCH::ClickUrl;
							contextAnalysisSearchResult.displayURL = resultNode.NAMESPACE_SEARCH::DisplayUrl;
							
							contextAnalysisSearchResult.mimeType = resultNode.NAMESPACE_SEARCH::MimeType;
							
							contextAnalysisSearchResult.cache = new Cache(resultNode.NAMESPACE_SEARCH::Cache.NAMESPACE_SEARCH::Url, 
								resultNode.NAMESPACE_SEARCH::Cache.NAMESPACE_SEARCH::Size as uint);
							
							//convert string to number to date
							contextAnalysisSearchResult.modificationDate = new Date(resultNode.NAMESPACE_SEARCH::ModificationDate as Number);
						
							/* The following are not implemented in the current XML API */
							//contextAnalysisSearchResult.category 
							//contextAnalysisSearchResult.internalSearchURL "More fromt this site" link
							
							// Add the WebSearchResult Object to our Data Array.
							resultsArray.push(contextAnalysisSearchResult);
							
						}
					
				break;
				
				case SearchResultEvent.RELATED_SUGGESTION_RESULT:
					
				    for each (resultNode in inXML.children())
					{
						var relatedSuggestionSearchResult:SearchResult = new SearchResult();
				
						// SearchResult Properties
						relatedSuggestionSearchResult.index = resultNode.childIndex() + len;
						relatedSuggestionSearchResult.name = resultNode.toString();
											
						// Add the SearchResult Object to our Data Array.
						resultsArray.push(relatedSuggestionSearchResult);
						
					}
				break;
				
				case SearchResultEvent.SPELLING_SUGGESTION_RESULT:
					
				    for each (resultNode in inXML.children())
					{
						var spellingSearchResult:SearchResult = new SearchResult();
			
						// SearchResult Properties
						spellingSearchResult.index = resultNode.childIndex() + len;
						spellingSearchResult.name = resultNode.toString();
					
						// Add the SearchResult Object to our Data Array.
						resultsArray.push(spellingSearchResult);
						
					}
					
				break;
				
				case SearchResultEvent.SONG_DOWNLOAD_LOCATION_SEARCH_RESULT:
					
				    for each (resultNode in inXML.children())
					{
						var songDownloadSearchResult:MediaSearchResult = new MediaSearchResult();
						
						// SearchResult Properties
						songDownloadSearchResult.index = resultNode.childIndex() + len;
						songDownloadSearchResult.name = resultNode.NAMESPACE_AUDIO_ALT::Url;
							
						songDownloadSearchResult.url = resultNode.NAMESPACE_AUDIO_ALT::Url;
						
						songDownloadSearchResult.publisher = resultNode.NAMESPACE_AUDIO_ALT::Source;
						
						songDownloadSearchResult.fileFormat = resultNode.NAMESPACE_AUDIO_ALT::Format;
						
						songDownloadSearchResult.restrictions = resultNode.NAMESPACE_AUDIO_ALT::Restrictions;
						
						songDownloadSearchResult.duration = resultNode.NAMESPACE_AUDIO_ALT::Length;
						songDownloadSearchResult.quality = resultNode.NAMESPACE_AUDIO_ALT::Quality;
						songDownloadSearchResult.price = resultNode.NAMESPACE_AUDIO_ALT::Price;
						songDownloadSearchResult.numChannels = resultNode.NAMESPACE_AUDIO_ALT::Channels;
					
						// Add the SearchResult Object to our Data Array.
						resultsArray.push(songDownloadSearchResult);
						
					}
					
				break;
				
				case SearchResultEvent.SONG_SEARCH_RESULT:
					
				    for each (resultNode in inXML.children())
					{
						var songSearchResult:Song = new Song();
						
						
						// SearchResult Properties
						songSearchResult.index = resultNode.NAMESPACE_AUDIO_ALT::Track;
						songSearchResult.name = resultNode.NAMESPACE_AUDIO_ALT::Title;
						songSearchResult.id = resultNode.@id;
						
						songSearchResult.album = new Album();
						songSearchResult.album.name = resultNode.NAMESPACE_AUDIO_ALT::Album;
						songSearchResult.album.id = resultNode.NAMESPACE_AUDIO_ALT::Album.@id;
						
						songSearchResult.duration = resultNode.NAMESPACE_AUDIO_ALT::Length;
						
						songSearchResult.publisher = resultNode.NAMESPACE_AUDIO_ALT::Publisher;
						
						songSearchResult.thumbnail = new Thumbnail(resultNode.NAMESPACE_AUDIO_ALT::Thumbnail.NAMESPACE_AUDIO_ALT::Url);
						songSearchResult.thumbnail.height = resultNode.NAMESPACE_AUDIO_ALT::Thumbnail.NAMESPACE_AUDIO_ALT::Height as Number;
						songSearchResult.thumbnail.width = resultNode.NAMESPACE_AUDIO_ALT::Thumbnail.NAMESPACE_AUDIO_ALT::Width as Number;
						
						songSearchResult.releaseDate = resultNode.NAMESPACE_AUDIO_ALT::ReleaseDate;
						
						// Add the SearchResult Object to our Data Array.
						resultsArray.push(songSearchResult);
						
					}
				
				break;
				
				case SearchResultEvent.ALBUM_SEARCH_RESULT:
					
				    for each (resultNode in inXML.children())
					{
						var albumSearchResult:AlbumSearchResult = new AlbumSearchResult();

						// SearchResult Properties
						albumSearchResult.index = resultNode.childIndex() + len;
						albumSearchResult.name = resultNode.NAMESPACE_AUDIO_ALT::Title;
						albumSearchResult.id = resultNode.@id;
						
						albumSearchResult.artist = new Artist();
						albumSearchResult.artist.name = resultNode.NAMESPACE_AUDIO_ALT::Artist;
						albumSearchResult.artist.id = resultNode.NAMESPACE_AUDIO_ALT::Artist.@id;
						
						albumSearchResult.publisher = resultNode.NAMESPACE_AUDIO_ALT::Publisher;
						
						albumSearchResult.thumbnail = new Thumbnail(resultNode.NAMESPACE_AUDIO_ALT::Thumbnail.NAMESPACE_AUDIO_ALT::Url);
						albumSearchResult.thumbnail.height = resultNode.NAMESPACE_AUDIO_ALT::Thumbnail.NAMESPACE_AUDIO_ALT::Height as Number;
						albumSearchResult.thumbnail.width = resultNode.NAMESPACE_AUDIO_ALT::Thumbnail.NAMESPACE_AUDIO_ALT::Width as Number;
						
						albumSearchResult.releaseDate = resultNode.NAMESPACE_AUDIO_ALT::ReleaseDate;
						
						albumSearchResult.relatedAlbums = [];
						
						for each(var a:XML in resultNode.NAMESPACE_AUDIO_ALT::RelatedAlbums.NAMESPACE_AUDIO_ALT::Album)
						{
							var album:Album = new Album();
							album.name = a.toString();
							album.id = a.@id;
							
							albumSearchResult.relatedAlbums.push(album);
						}
						
						// Add the SearchResult Object to our Data Array.
						resultsArray.push(albumSearchResult);
					}
				break;
				
				
				case SearchResultEvent.ARTIST_SEARCH_RESULT:
					
				    for each (resultNode in inXML.children())
					{
						var artistSearchResult:ArtistSearchResult = new ArtistSearchResult();
						
						// SearchResult Properties
						artistSearchResult.index = resultNode.childIndex() + len;
						artistSearchResult.name = resultNode.NAMESPACE_AUDIO_ALT::Name;
							
						artistSearchResult.url = resultNode.NAMESPACE_AUDIO_ALT::YahooMusicPage;
						
						artistSearchResult.thumbnail = new Thumbnail(resultNode.NAMESPACE_AUDIO_ALT::Thumbnail.NAMESPACE_AUDIO_ALT::Url);
						artistSearchResult.thumbnail.height = resultNode.NAMESPACE_AUDIO_ALT::Thumbnail.NAMESPACE_AUDIO_ALT::Height as Number;
						artistSearchResult.thumbnail.width = resultNode.NAMESPACE_AUDIO_ALT::Thumbnail.NAMESPACE_AUDIO_ALT::Width as Number;
						
						artistSearchResult.relatedArtists = [];
						
						for each(var aXML:XML in resultNode.NAMESPACE_AUDIO_ALT::RelatedArtists.NAMESPACE_AUDIO_ALT::Artist)
						{
							var artist:Artist = new Artist();
							artist.name = aXML.toString();
							artist.id = aXML.@id;
							
							artistSearchResult.relatedArtists.push(artist);
						}
					
						artistSearchResult.popularSongs = [];
						
						for each(var s:XML in resultNode.NAMESPACE_AUDIO_ALT::PopularSongs.NAMESPACE_AUDIO_ALT::Song)
						{
							var song:Song = new Song();
							song.name = s.toString();
							song.id = s.@id;
							
							artistSearchResult.popularSongs.push(song);
						}
						
						// Add the SearchResult Object to our Data Array.
						resultsArray.push(artistSearchResult);
					}
				break;
				
				case SearchResultEvent.AUDIO_SEARCH_RESULT:
					
				    for each (resultNode in inXML.children())
					{
						var audioSearchResult:MediaSearchResult = new MediaSearchResult();
					
						// SearchResult Properties
						audioSearchResult.index = resultNode.childIndex() + len;
						audioSearchResult.name = resultNode.NAMESPACE_AUDIO::Title;
						if(audioSearchResult.name == "") audioSearchResult.name = resultNode.NAMESPACE_AUDIO::Url;
							
						audioSearchResult.url = resultNode.NAMESPACE_AUDIO::Url;
						audioSearchResult.clickURL = resultNode.NAMESPACE_AUDIO::ClickUrl;
						audioSearchResult.referrerURL = resultNode.NAMESPACE_AUDIO::ReferrerUrl;
						
						audioSearchResult.isStreaming = resultNode.NAMESPACE_AUDIO::Streaming as Boolean;
						audioSearchResult.duration = resultNode.NAMESPACE_AUDIO::Duration;
						audioSearchResult.numChannels = resultNode.NAMESPACE_AUDIO::Channels;
					
						// Add the SearchResult Object to our Data Array.
						resultsArray.push(audioSearchResult);
					}
				break;
				
				case SearchResultEvent.PODCAST_SEARCH_RESULT:
					
				    for each (resultNode in inXML.children())
					{
						var podcastSearchResult:MediaSearchResult = new MediaSearchResult();
						
						// SearchResult Properties
						podcastSearchResult.index = resultNode.childIndex() + len;
						podcastSearchResult.name = resultNode.NAMESPACE_SEARCH::Title;

						podcastSearchResult.url = resultNode.NAMESPACE_SEARCH::Url;
						podcastSearchResult.clickURL = resultNode.NAMESPACE_SEARCH::ClickUrl;
						podcastSearchResult.referrerURL = resultNode.NAMESPACE_SEARCH::ReferrerUrl;
						
						podcastSearchResult.isStreaming = resultNode.NAMESPACE_SEARCH::Streaming as Boolean;
						
						podcastSearchResult.duration = resultNode.NAMESPACE_SEARCH::Duration;
						podcastSearchResult.numChannels = resultNode.NAMESPACE_SEARCH::Channels;
					
					
						// Add the SearchResult Object to our Data Array.
						resultsArray.push(podcastSearchResult);
						
					}
				break;
				
				case SearchResultEvent.VIDEO_SEARCH_RESULT:
					
				    for each (resultNode in inXML.children())
					{
						var videoSearchResult:MediaSearchResult = new MediaSearchResult();
						
						// SearchResult Properties
						videoSearchResult.index = resultNode.childIndex() + len;
						videoSearchResult.name = resultNode.NAMESPACE_VIDEO::Title;

						videoSearchResult.url = resultNode.NAMESPACE_VIDEO::Url;
						videoSearchResult.clickURL = resultNode.NAMESPACE_VIDEO::ClickUrl;
						videoSearchResult.referrerURL = resultNode.NAMESPACE_VIDEO::ReferrerUrl;
						
						videoSearchResult.thumbnail = new Thumbnail(resultNode.NAMESPACE_VIDEO::Thumbnail.NAMESPACE_VIDEO::Url);
						videoSearchResult.thumbnail.height = resultNode.NAMESPACE_VIDEO::Thumbnail.NAMESPACE_VIDEO::Height;
						videoSearchResult.thumbnail.width = resultNode.NAMESPACE_VIDEO::Thumbnail.NAMESPACE_VIDEO::Width;
						
						videoSearchResult.isStreaming = resultNode.NAMESPACE_VIDEO::Streaming as Boolean;
						
						videoSearchResult.duration = resultNode.NAMESPACE_VIDEO::Duration;
						videoSearchResult.numChannels = resultNode.NAMESPACE_VIDEO::Channels;
					
						
						// Add the SearchResult Object to our Data Array.
						resultsArray.push(videoSearchResult);
					}
				break;
				
				case SearchResultEvent.NEWS_SEARCH_RESULT:
					
				    for each (resultNode in inXML.children())
					{
						var newsSearchResult:NewsSearchResult = new NewsSearchResult();
						
						
						// SearchResult Properties
						newsSearchResult.index = resultNode.childIndex() + len;
						newsSearchResult.name = resultNode.NAMESPACE_NEWS::Title;
						newsSearchResult.language = resultNode.NAMESPACE_NEWS::Language;
						
						newsSearchResult.url = resultNode.NAMESPACE_NEWS::Url;
					
						newsSearchResult.clickURL = resultNode.NAMESPACE_NEWS::ClickUrl;
						newsSearchResult.sourceURL = resultNode.NAMESPACE_NEWS::NewsSourceURL;
					
						newsSearchResult.newsSource = resultNode.NAMESPACE_NEWS::NewsSource;
						
						//convert string to number to date
						newsSearchResult.modificationDate = new Date(resultNode.NAMESPACE_NEWS::ModificationDate as Number);
						newsSearchResult.publicationDate = new Date(resultNode.NAMESPACE_NEWS::PublicationDate as Number);
					
						// Add the SearchResult Object to our Data Array.
						resultsArray.push(newsSearchResult);	
					}
				break;
				
				case SearchResultEvent.IMAGE_SEARCH_RESULT:
					
				    for each (resultNode in inXML.children())
					{
						var imageSearchResult:MediaSearchResult = new MediaSearchResult();
						
						
						// SearchResult Properties
						imageSearchResult.index = resultNode.childIndex() + len;
						imageSearchResult.name = resultNode.NAMESPACE_IMAGE::Title;
						imageSearchResult.summary = resultNode.NAMESPACE_IMAGE::Summary;
						
						imageSearchResult.url = resultNode.NAMESPACE_IMAGE::Url;
						imageSearchResult.thumbnail = new Thumbnail(resultNode.NAMESPACE_IMAGE::Thumbnail.NAMESPACE_IMAGE::Url);
						imageSearchResult.thumbnail.height = resultNode.NAMESPACE_IMAGE::Thumbnail.NAMESPACE_IMAGE::Height as Number;
						imageSearchResult.thumbnail.width = resultNode.NAMESPACE_IMAGE::Thumbnail.NAMESPACE_IMAGE::Width as Number;
						
						imageSearchResult.clickURL = resultNode.NAMESPACE_IMAGE::ClickUrl;
						imageSearchResult.referrerURL = resultNode.NAMESPACE_IMAGE::ReferrerUrl;
						
						imageSearchResult.fileSize = resultNode.NAMESPACE_IMAGE::FileSize as uint;
						imageSearchResult.fileFormat = resultNode.NAMESPACE_IMAGE::FileFormat;
					
						imageSearchResult.restrictions = resultNode.NAMESPACE_IMAGE::Restrictions;
						imageSearchResult.publisher = resultNode.NAMESPACE_IMAGE::Publisher;
						imageSearchResult.copyright = resultNode.NAMESPACE_IMAGE::Copyright;
						
						imageSearchResult.height = resultNode.NAMESPACE_IMAGE::Height;
						imageSearchResult.width = resultNode.NAMESPACE_IMAGE::Width;
						
						// Add the SearchResult Object to our Data Array.
						resultsArray.push(imageSearchResult);	
					}
				break;
				
				//default web search
				default:
				    for each (resultNode in inXML.children())
					{
						var resultObject:WebSearchResult = new WebSearchResult();
						
						
						// WebSearchResult Properties
						//index is the current index in this set of results added to what has already been returned, if anything
						resultObject.index = resultNode.childIndex() + len;
						resultObject.name = resultNode.NAMESPACE_SEARCH::Title;
						resultObject.summary = resultNode.NAMESPACE_SEARCH::Summary;
						
						resultObject.url = resultNode.NAMESPACE_SEARCH::Url;
						resultObject.clickURL = resultNode.NAMESPACE_SEARCH::ClickUrl;
						resultObject.displayURL = resultNode.NAMESPACE_SEARCH::DisplayUrl;
						
						resultObject.mimeType = resultNode.NAMESPACE_SEARCH::MimeType;
						
						resultObject.cache = new Cache(resultNode.NAMESPACE_SEARCH::Cache.NAMESPACE_SEARCH::Url, 
							resultNode.NAMESPACE_SEARCH::Cache.NAMESPACE_SEARCH::Size as uint);
						
						//convert string to number to date
						resultObject.modificationDate = new Date(resultNode.NAMESPACE_SEARCH::ModificationDate as Number);
					
						/* The following are not implemented in the current XML API */
						//resultObject.category 
						//resultObject.internalSearchURL "More fromt this site" link
						
						// Add the WebSearchResult Object to our Data Array.
						resultsArray.push(resultObject);
					}
			}
		   
			//if this search is being continued, add these to what we already have
			if(increment > 0)
			{
				lastResult = lastResult.concat(resultsArray);
			}
			else
			{
				lastResult = resultsArray;
			}
			
			
			if(firstResultPosition +  lastResult.length >= numResultsAvailable)
			{
				hasResultsPending = false;
				
			}
						
			//Dispatch result
			dispatchResult(resultsArray, dispatchType);
		}
		
		
		/**
		* @private
		* Internal method for dispatching Events.
		*
		* @param result The Object containing the result
		* @param dispatchType The Event Type
		*/
		protected function dispatchResult(result:Object, dispatchType:String):void
		{
			//Dispatch a general Result event, passing along the type of result it is
			var aResults:SearchResultEvent = new SearchResultEvent(SearchResultEvent.RESULT, result, dispatchType);
			
			dispatchEvent(aResults);
		}
		
		
		/**
		* @private
		* Internal method for dispatching Fault Events.
		*
		* @param fault The Fault
		*/
		protected function dispatchFault(fault:com.yahoo.webapis.ServiceFault):void
		{
			var ServiceFault:SearchFaultEvent = new SearchFaultEvent(SearchFaultEvent.FAULT, fault);
			dispatchEvent(ServiceFault);
			
		}
		
		
	}
}