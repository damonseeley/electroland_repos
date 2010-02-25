/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.search
{
	/**
	 * WebSearchResult is a Value Object for the Search API.
	 * @langversion ActionScript 3.0
	 * @playerversion Flash 9
	 * @author Alaric Cole 02/22/07
	 */
	 
	public class WebSearchResult extends SearchResult
	{
		
		private var _summary:String;
		
		private var _mimeType:String;
		
		private var _category:String;
		private var _categories:Array;
		
		private var _url:String;
		private var _clickURL:String;
		private var _displayURL:String;
		
		private var _modificationDate:Date;
		
		private var _cache:Cache;

		
		/**
		 * Construct a new WebSearchResult instance.
		 */
		public function WebSearchResult()
		{
			// Do Nothing.
		}
		
		
		/**
		 * The Summary.
		 */
		public function get summary():String
		{
			return _summary;
		}
		
		public function set summary( value:String ):void
		{
			_summary = value;
		}
		
		/**
		 * The mime Type.
		 */
		public function get mimeType():String
		{
			return _mimeType;
		}
		
		public function set mimeType( value:String ):void
		{
			_mimeType = value;
		}
		
		
		/**
		 * The Yahoo! category.
		 */
		public function get category():String
		{
			return _category;
		}
		
		public function set category( value:String ):void
		{
			_category = value;
		}
		
		
		/**
		 * The categories.
		 */
		public function get categories():Array
		{
			return _categories;
		}
		
		public function set categories( value:Array ):void
		{
			_categories = value;
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
		 * The display URL.
		 */
		public function get displayURL():String
		{
			return _displayURL;
		}
		
		public function set displayURL( value:String ):void
		{
			_displayURL = value;
		}
		
		/**
		 * The Modification Date.
		 */
		public function get modificationDate():Date
		{
			return _modificationDate;
		}
		
		public function set modificationDate( value:Date ):void
		{
			_modificationDate = value;
		}
		
		
		/**
		 * The cache.
		 */
		public function get cache():Cache
		{
			return _cache;
		}
		
		public function set cache( value:Cache ):void
		{
			_cache = value;
		}
		
	}
}