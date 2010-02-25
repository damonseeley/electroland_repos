/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.search.params
{
	import flash.net.URLLoader;
	
	/**
	* An object for passing parameters to the search.
	* 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Alaric Cole 02/22/07
	*/
	public class SearchParams
	{
		private var _query:String;
		private var _start:uint;
		private var _site:String;
		private var _maximumResults:uint;
		private var _searchIn:String;
		private var _license:String;
		private var _allowAdultContent:Boolean;
		private var _allowSimilarContent:Boolean;
		private var _language:String;
		private var _context:String;
		private var _type:String;
		private var _artist:String;
		private var _artistId:String;
		private var _album:String;
		private var _albumId:String;
		private var _song:String;
		private var _songId:String;
		private var _source:String;
		private var _coloration:String;
		private var _listingId:String;
		private var _radius:String;
		private var _street:String;
		private var _city:String;
		private var _state:String;
		private var _zipCode:String;
		private var _location:String;
		private var _latitude:Number;
		private var _longitude:Number;
		private var _omitCategory:uint;
		private var _minimumRating:uint;
		
		private var _format:String;
		private var _category:uint;
		private var _country:String;
		private var _dateRange:String;
		private var _sort:String;
		
		
		/**
		* SearchParams	Constructor
		*/
		public function SearchParams()
		{
			//do nothing
		}
		
		/**
		* Returns a string containing all enumerable variables, in the MIME content encoding application/x-www-form-urlencoded. 
		*/
		public function collect():String
		{
			var resultsList:Array = new Array();
			
			if(query) resultsList.push("query=" + escape( query ));
			
			if(listingId) resultsList.push("listing_id=" + escape( listingId ));
			if(radius) resultsList.push("radius=" + escape( radius ));
			if(street) resultsList.push("street=" + escape( street ));
			if(city) resultsList.push("city=" + escape( city ));
			if(state) resultsList.push("state=" + escape( state ));
			if(zipCode) resultsList.push("zip=" + escape( zipCode ));
			if(location) resultsList.push("location=" + escape( location ));
			if(latitude) resultsList.push("latitude=" + escape( latitude.toString() ));
			if(longitude) resultsList.push("longitude=" + escape( longitude.toString() ));
			if(omitCategory) resultsList.push("omit_category=" + escape( omitCategory.toString() ));
			if(minimumRating) resultsList.push("minimum_rating=" + escape( minimumRating.toString() ));

			if(type) resultsList.push("type=" + escape( type ));
			if(artist) resultsList.push("artist=" + escape( artist ));
			if(artistId) resultsList.push("artistid=" + escape( artistId ));
			if(album) resultsList.push("album=" + escape( album ));
			if(albumId) resultsList.push("albumid=" + escape( albumId ));
			if(song) resultsList.push("song=" + escape( song ));
			if(songId) resultsList.push("songid=" + escape( songId ));
			if(source) resultsList.push("source=" + escape( source ));
			if(coloration) resultsList.push("coloration=" + escape( coloration ));
			if(license) resultsList.push("license=" + escape( license ));
			if(allowAdultContent) resultsList.push("adult_ok=" + "1");
			if(allowSimilarContent) resultsList.push("similar_ok=" + "1");
			if(language) resultsList.push("language=" + escape( language ));
			if(format) resultsList.push("format=" + escape( format ));
			if(searchIn) resultsList.push("search_in=" + escape( searchIn ));
			if(category) resultsList.push("category=" + escape( category.toString() ));
			if(country) resultsList.push("country=" + escape( country ));
			if(dateRange) resultsList.push("date_range=" + escape( dateRange ));
			if(sort) resultsList.push("sort=" + escape( sort ));
			if(start) resultsList.push("start=" + escape( start.toString() ));
			if(maximumResults) resultsList.push("results=" + escape( maximumResults.toString()));
			if(context) resultsList.push("context=" + escape( context ));
			if(site) resultsList.push("site=" + escape( site ));
			
			var returnString:String = "";
			for(var i:Number = 0; i < resultsList.length; i++)
			{
				var item:String = resultsList[i];
				returnString += item;
				
				if(i < resultsList.length-1)
				{
					returnString += "&";
				}
			}
			return returnString;
		}
		
			
		
		/**
		 * The search terms to look for. Use + to include terms, - to exclude terms, and put quotes around "exact phrase".
		 */
		public function get query():String
		{
			return _query;
		}
		
		public function set query( value:String ):void
		{
			_query = value;
		}
		
		/**
		 * The id associated with a specific business listing. 
		 * It corresponds with the id attribute of Result entities. At least one of query or listing id must be specified.
		 */
		public function get listingId():String
		{
			return _listingId;
		}
		
		public function set listingId( value:String ):void
		{
			_listingId = value;
		}
		
		/**
		* The source of the download. You may specify multiple values and a file from any of the sources specified will be returned. 
		* Valid values are:
		* audiolunchbox
	    * artistdirect
	    * buymusic
	    * dmusic
	    * emusic
	    * epitonic
	    * garageband
	    * itunes
	    * yahoo
	    * livedownloads
	    * mp34u
	    * msn
	    * musicmatch
	    * mapster
	    * passalong
	    * rhapsody
	    * soundclick
	    * theweb
		*/
		public function get source():String
		{
			return _source;
		}
		
		public function set source( value:String ):void
		{
			_source = value;
		}
		
		/**
		 *  How far (in miles) from the specified location to search for the query terms. The default radius varies according to the location given.
		 */
		public function get radius():String
		{
			return _radius;
		}
		
		public function set radius( value:String ):void
		{
			_radius = value;
		}
		
		/**
		 *  The service returns only the images of the coloration specified (color or black-and-white).
		 */
		public function get coloration():String
		{
			return _coloration;
		}
		
		public function set coloration( value:String ):void
		{
			_coloration = value;
		}
		
		/**
		 *  Street name. The number is optional.
		 */
		public function get street():String
		{
			return _street;
		}
		
		public function set street( value:String ):void
		{
			_street = value;
		}
		/**
		 *  The city name.
		 */
		public function get city():String
		{
			return _city;
		}
		
		public function set city( value:String ):void
		{
			_city = value;
		}
		/**
		 *  The starting result position to return (1-based).
		 */
		public function get state():String
		{
			return _state;
		}
		
		public function set state( value:String ):void
		{
			_state = value;
		}
		/**
		 * zip  The five-digit zip code, or the five-digit code plus four-digit extension. If this location contradicts the city and state specified, the zip code will be used for determining the location and the city and state will be ignored.
		 */
		public function get zipCode():String
		{
			return _zipCode;
		}
		
		public function set zipCode( value:String ):void
		{
			_zipCode = value;
		}
		
		/**
		* This free field lets users enter any of the following:
		* city, state
	    * city, state, zip
	    * zip
	    * street, city, state
	    * street, city, state, zip
	    * street, zip
	    * If location is specified, it will take priority over the individual fields in determining the location for the query. City, state and zip will be ignored.
		*/
		public function get location():String
		{
			return _location;
		}
		
		public function set location( value:String ):void
		{
			_location = value;
		}
		/**
		 *  The latitude of the starting location.
		 */
		public function get latitude():Number
		{
			return _latitude;
		}
		
		public function set latitude( value:Number ):void
		{
			_latitude = value;
		}
		/**
		 *  The longitude of the starting location. If both latitude and longitude are specified, they will take priority over all other location data. 
		 * If only one of latitude or longitude is specified, both will be ignored.
		 */
		public function get longitude():Number
		{
			return _longitude;
		}
		
		public function set longitude( value:Number ):void
		{
			_longitude = value;
		}
		/**
		 * The id of a category to omit results from. 
		 * Multiple categories may be omitted (omit_category=1234&omit_category=5678), and a result will not be returned if it appears in any of the specified categories.
		 */
		public function get omitCategory():uint
		{
			return _omitCategory;
		}
		
		public function set omitCategory( value:uint ):void
		{
			_omitCategory = value;
		}
		
		/**
		 * The minimum average rating (on a five point scale) for a result. If this is specified, no results without ratings will be returned.
		 */
		public function get minimumRating():uint
		{
			return _minimumRating;
		}
		
		public function set minimumRating( value:uint ):void
		{
			_minimumRating = value;
		}
		
		
		/**
		 *  The type of search to submit.
		 */
		public function get type():String
		{
			return _type;
		}
		
		public function set type( value:String ):void
		{
			_type = value;
		}
		
		/**
		 *  The artist terms to look for.
		 */
		public function get artist():String
		{
			return _artist;
		}
		
		public function set artist( value:String ):void
		{
			_artist = value;
		}
		
		/**
		 *  The search terms to look for.
		 */
		public function get artistId():String
		{
			return _artistId;
		}
		
		public function set artistId( value:String ):void
		{
			_artistId = value;
		}
		
		/**
		 *  The album name or partial album string to search for (UTF-8 encoded).
		 */
		public function get album():String
		{
			return _album;
		}
		
		public function set album( value:String ):void
		{
			_album = value;
		}
		
		/**
		 *  The specific id for an album. Ids are internal to the Music Search Service and will be returned with album references. At least one of artist, artistid, album or albumid is required.
		 */
		public function get albumId():String
		{
			return _albumId;
		}
		public function set albumId( value:String ):void
		{
			_albumId = value;
		}
		
		/**
		 *  The song title or partial song title string to search for (UTF-8 encoded).
		 */
		public function get song():String
		{
			return _song;
		}
		
		public function set song( value:String ):void
		{
			_song= value;
		}
		
		/**
		 *  The specific id for a song. Ids are internal to the Music Search Service and will be returned with song references. At least one of artist, artistid, album, albumid, song or songid is required.
		 */
		public function get songId():String
		{
			return _songId;
		}
		
		public function set songId( value:String ):void
		{
			_songId = value;
		}
		
		/**
		 *  Specifies the kind of file to search for. html, msword, pdf, ppt, rss, txt, xls
		 * @default all
		 */
		public function get format():String
		{
			return _format;
		}
		
		public function set format( value:String ):void
		{
			_format = value;
		}
		
		/**
		 *  The context to extract meaning from (UTF-8 encoded).
		 */
		public function get context():String
		{
			return _context;
		}
		
		public function set context( value:String ):void
		{
			_context = value;
		}
		
		/**
		 * The language the results are written in.
		 * @see http://developer.yahoo.com/search/languages.html
		 */
		public function get language():String
		{
			return _language;
		}
		
		public function set language( value:String ):void
		{
			_language = value;
		}
		
		
		
		/**
		 *  Specifies whether to allow multiple results with similar content. Enter a 1 to allow similar content.
		 */
		public function get allowSimilarContent():Boolean
		{
			return _allowSimilarContent;
		}
		
		public function set allowSimilarContent( value:Boolean ):void
		{
			_allowSimilarContent = value;
		}
		
	
		/**
		 *  Specifies whether to allow results with adult content. Enter a 1 to allow adult content.
		 */
		public function get allowAdultContent():Boolean
		{
			return _allowAdultContent;
		}
		
		public function set allowAdultContent( value:Boolean ):void
		{
			_allowAdultContent = value;
		}
		
		
		/**
		 *  The Creative Commons license that the contents are licensed under. cc_any, cc_commercial, cc_modifiable
		 * @default all
		 */
		public function get license():String
		{
			return _license;
		}
		
		public function set license( value:String ):void
		{
			_license = value;
		}
		
		
		/**
		 *  The site to limit the search to.
		 */
		public function get site():String
		{
			return _site;
		}
		
		public function set site( value:String ):void
		{
			_site = value;
		}
		
		/**
		 *  Search for keywords. When ommited default is "all".
		 */
		public function get searchIn():String
		{
			return _searchIn;
		}
		
		public function set searchIn( value:String ):void
		{
			_searchIn = value;
		}

		
		/**
		 *  Search only in the specified category name or names. Will match against the full path to the English category name as found on the Yahoo! Search site. Category names are case-sensitive and should be URL-encoded. Computers & Internet>Software, for example, looks like this: Computers+%26+Internet%3ESoftware.
		 */
		public function get category():uint
		{
			return _category;
		}
		
		public function set category( value:uint ):void
		{
			_category = value;
		}
		
		
		
		/**
		 * Filter based on country: You can enter multiple countrys. The first will determine the destination country of hyperlinks.
		 * @see http://developer.yahoo.com/search/countries.html
		 */
		public function get country():String
		{
			return _country;
		}
		
		public function set country( value:String ):void
		{
			_country = value;
		}
		
		
		
		/**
		* Filter based on date submitted:
		* all: Anytime
	    * 7: Within 7 days
	    * 7-30: Within 7-30 days
	    * 30-60: Within 30-60 days
	    * 60-90: Within 60-90 days
	    * more90: More than 90 days
	    * 
	    * @default all
		*/
		public function get dateRange():String
		{
			return _dateRange;
		}
		
		public function set dateRange( value:String ):void
		{
			_dateRange = value;
		}
		
		
		/**
		* Sorting order of result set:
		* relevance: By relevance.
	    * title: By the page title.
	    * distance: By distance (local only).
	    * rating: By rating (local only)
	    * 
	    * @default relevance
		*/
		public function get sort():String
		{
			return _sort;
		}
		
		public function set sort( value:String ):void
		{
			_sort = value;
		}
		
		
		/**
		 *  Starting page to list, used to display further results. default 0
		 */
		public function get start():uint
		{
			return _start;
		}
		
		public function set start( value:uint ):void
		{
			_start = value;
		}
		
		
		/**
		 * Number of pages to be returned. Between 1 and 100
		 * @default 10
		 */
		public function get maximumResults():uint
		{
			return _maximumResults;
		}
		
		public function set maximumResults( value:uint ):void
		{
			if(value > 100) value = 100;
			_maximumResults = value;
		}
		
		/**
		 * The params as a string
		 */
		public function toString():String
		{
			return collect();
		}
	}
}