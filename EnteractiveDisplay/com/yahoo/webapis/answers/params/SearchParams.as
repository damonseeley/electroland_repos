/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.answers.params
{
	import flash.net.URLLoader;
	
	/**
	* Yahoo! Answers API - A Object for Passing Parametors to the Query Search, uses URLLoader through composition.
	* 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Hepp Maccoy 01/24/07
	*/
	public class SearchParams
	{
		private var _search_in:String;
		private var _category_id:String;
		private var _category_name:String;
		private var _region:String;
		private var _date_range:String;
		private var _sort:String;
		private var _type:String;
		private var _start:Number;
		private var _results:Number;
		
		
		/**
		* SearchParams	Constructor
		*/
		public function SearchParams()
		{
			//do nothing
		}
		
		/**
		* collect	Returns a string containing all enumerable variables, in the MIME content encoding application/x-www-form-urlencoded. 
		*/
		public function collect():String
		{
			var resultsList:Array = new Array();
			
			if(search_in != null){resultsList.push("search_in=" + search_in)};
			if(category_id != null){resultsList.push("category_id=" + category_id)};
			if(category_name != null){resultsList.push("category_name=" + category_name)};
			if(region != null){resultsList.push("region=" + region)};
			if(date_range != null){resultsList.push("date_range=" + date_range)};
			if(sort != null){resultsList.push("sort=" + sort)};
			if(type != null){resultsList.push("type=" + type)};
			if(!isNaN(start)){resultsList.push("start=" + String(start))};
			if(!isNaN(results)){resultsList.push("results=" + String(results))};
			
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
		 * search_in - Search for keywords in: "all", "question", "best_answer", "nickname". When ommited default is "all".
		 */
		public function get search_in():String
		{
			return _search_in;
		}
		
		public function set search_in( value:String ):void
		{
			_search_in = escape(value);
		}
		
		
		
		/**
		 * category_id - Search only in the specified category ID or IDs. (IDs may be seen in the request URLs when browsing Yahoo! Answers categories.)
		 */
		public function get category_id():String
		{
			return _category_id;
		}
		
		public function set category_id( value:String ):void
		{
			_category_id = escape(value);
		}
		
		
		
		/**
		 * category_name - Search only in the specified category name or names. Will match against the full path to the English category name as found on the Yahoo! Answers site. Category names are case-sensitive and should be URL-encoded. Computers & Internet>Software, for example, looks like this: Computers+%26+Internet%3ESoftware. omit for default to all categories
		 */
		public function get category_name():String
		{
			return _category_name;
		}
		
		public function set category_name( value:String ):void
		{
			_category_name = escape(value);
		}
		
		
		
		/**
		 * region - Filter based on country:

		    * us: United States
		    * uk: United Kingdom
		    * ca: Canada
		    * au: Australia
		    * in: India
		    * es: Spain
		    * br: Brazil
		    * ar: Argentina
		    * mx: Mexico
		    * e1: en Espanol
		    * it: Italy
		    * de: Germany
		    * fr: France
		    * sg: Singapore
		    * 
		    * omit for default "us".
		
		    You can enter multiple regions. The first will determine the destination country of hyperlinks.
		 */
		public function get region():String
		{
			return _region;
		}
		
		public function set region( value:String ):void
		{
			_region = escape(value);
		}
		
		
		
		/**
		 * date_range - Filter based on date submitted:

		    * all: Anytime
		    * 7: Within 7 days
		    * 7-30: Within 7-30 days
		    * 30-60: Within 30-60 days
		    * 60-90: Within 60-90 days
		    * more90: More than 90 days
		    * 
		    * omit for default "all".
		 */
		public function get date_range():String
		{
			return _date_range;
		}
		
		public function set date_range( value:String ):void
		{
			_date_range = escape(value);
		}
		
		
		
		/**
		 * sort - Sorting order of result set:

		    * relevance: By relevance.
		    * date_desc: By date, newest first.
		    * date_asc: By date, oldest first.
		    * 
		    * omit for default "relevance".

		 */
		public function get sort():String
		{
			return _sort;
		}
		
		public function set sort( value:String ):void
		{
			_sort = escape(value);
		}
		
		
		
		/**
		 * type - Question status. Set to "all", "resolved", "open", or "undecided". omit for default "all"
		 */
		public function get type():String
		{
			return _type;
		}
		
		public function set type( value:String ):void
		{
			_type = escape(value);
		}
		
		
		
		/**
		 * start - Starting question to list, used to display further results. default 0
		 */
		public function get start():Number
		{
			return _start;
		}
		
		public function set start( value:Number ):void
		{
			_start = value;
		}
		
		
		
		/**
		 * results - Number of questions to be returned. default 10, max 50
		 */
		public function get results():Number
		{
			return _results;
		}
		
		public function set results( value:Number ):void
		{
			_results = value;
		}
	}
}