/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.search
{
	
	/**
	 * NewsSearchResult is a Value Object for the Search API.
 	 * @langversion ActionScript 3.0
	 * @playerversion Flash 9
	 * @author Alaric Cole 02/22/07
	 */
	public class NewsSearchResult extends SearchResult
	{
		
		private var _newsSource:String;
		
		private var _language:String;
		
		private var _url:String;
		private var _clickURL:String;
		private var _sourceURL:String;
		
		private var _modificationDate:Date;
		private var _publicationDate:Date;
		
		
		/**
		 * Construct a new NewsSearchResult instance.
		 */
		public function NewsSearchResult()
		{
			// Do Nothing.
		}
		
		
		/**
		 * The new source, such as API or BBC.
		 */
		public function get newsSource():String
		{
			return _newsSource;
		}
		
		public function set newsSource( value:String ):void
		{
			_newsSource = value;
		}
		
		
		
		/**
		 * The language.
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
		 * The url.
		 */
		public function get url():String
		{
			return _url;
		}
		
		public function set url( value:String ):void
		{
			_url = value;
		}
		
		
		/**
		 * The click URL.
		 */
		public function get clickURL():String
		{
			return _clickURL;
		}
		
		public function set clickURL( value:String ):void
		{
			_clickURL = value;
		}
		
		/**
		 * The news source URL.
		 */
		public function get sourceURL():String
		{
			return _sourceURL;
		}
		
		public function set sourceURL( value:String ):void
		{
			_sourceURL = value;
		}
		
		
		/**
		 * The Date of publication.
		 */
		public function get publicationDate():Date
		{
			return _publicationDate;
		}
		
		public function set publicationDate( value:Date ):void
		{
			_publicationDate = value;
		}
		
		/**
		 * The Date.
		 */
		public function get modificationDate():Date
		{
			return _modificationDate;
		}
		
		public function set modificationDate( value:Date ):void
		{
			_modificationDate = value;
		}
		
		
		
	}
}