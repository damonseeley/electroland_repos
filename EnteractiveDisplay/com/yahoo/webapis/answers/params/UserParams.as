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
	public class UserParams
	{
		
		private var _sort:String;
		private var _type:String;
		private var _start:Number;
		private var _results:Number;
		private var _user_id:String;
		private var _filter:String;
		
		
		/**
		* UserParams	Constructor
		*/
		public function UserParams()
		{
			//do nothing
		}
		
		/**
		* collect	Returns a string containing all enumerable variables, in the MIME content encoding application/x-www-form-urlencoded. 
		*/
		public function collect():String
		{
			var resultsList:Array = new Array();
			
			if(sort != null){resultsList.push("sort=" + sort)};
			if(type != null){resultsList.push("type=" + type)};
			if(!isNaN(start)){resultsList.push("start=" + String(start))};
			if(!isNaN(results)){resultsList.push("results=" + String(results))};
			if(user_id != null){resultsList.push("user_id=" + user_id)};
			if(filter != null){resultsList.push("filter=" + filter)};
			
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
		
		
		
		/**
		 * user_id - User ID: unique per user. 
		 */
		public function get user_id():String
		{
			return _user_id;
		}
		
		public function set user_id( value:String ):void
		{
			_user_id = escape(value);
		}
		
		
		
		/**
		 * filter
		 * *
		 * Type of result set:
		 *
	     * question: Questions asked by the user.
	     * answer: Questions answered by the user.
    	 * best_answer: Questions where the user's answer was chosen as best answer. Only useful where type=resolved.
    	 * 
    	 * omit for default "question".
		 */
		public function get filter():String
		{
			return _filter;
		}
		
		public function set filter( value:String ):void
		{
			_filter = escape(value);
		}
	}
}