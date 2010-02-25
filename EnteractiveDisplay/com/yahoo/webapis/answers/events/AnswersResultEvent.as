/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.answers.events
{
	import flash.events.Event;
	
	/**
	* Event class in response to data events from the Yahoo! Answers API.
	* 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Hepp Maccoy 01/04/07
	*/
	public class AnswersResultEvent extends Event
	{
		/** Constant for the event types. */
		public static const SEARCH_BY_QUERY_RESULT:String = "searchByQueryResult";
		public static const SEARCH_BY_CATEGORY_RESULT:String = "searchByCategoryResult";
		public static const SEARCH_BY_USER_RESULT:String = "searchByUserResult";
		public static const GET_QUESTION_RESULT:String = "getQuestionResult";
		public static const GET_ANSWERS_RESULT:String = "getAnswersResult";
		public static const GET_COMMENTS_RESULT:String = "getCommentsResult";
		
		/**
		 * True if the event is the result of a successful call,
		 * False if the call failed
		 */
		public var success:Boolean;
		
		private var _data:Object;
		
		/**
		 * Constructs a new FlickrResultEvent
		 */
		public function AnswersResultEvent(type:String, inData:Object)
		{
			_data = inData;
									   	
			super( type, bubbles, cancelable );
		}
		
		
		/**
		 * Data
		 */
		public function get data():Object
		{
			return _data;
		}
		
		public function set data( value:Object ):void
		{
			_data = value;
		}
	
	}
	
}