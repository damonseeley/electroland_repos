/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.answers.events
{
	import flash.events.Event;
	
	/**
	* Error Event class in response to data events from the Yahoo! Answers API.
	* 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Hepp Maccoy 01/26/07
	*/
	public class AnswersErrorEvent extends Event
	{
		/** Constant for the event types. */
		public static const ERROR_EVENT:String = "errorEvent";
		
		/**
		 * True if the event is the result of a successful call,
		 * False if the call failed
		 */
		public var success:Boolean;
		
		private var _data:Object;
		private var _errorType:String;
		
		/**
		 * Constructs a new FlickrResultEvent
		 */
		public function AnswersErrorEvent(type:String, inData:Object, inErrorType:String)
		{
			_data = inData;
			_errorType = inErrorType;
									   	
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
		
		/**
		 * Error Type
		 * 
		 * Available errorType values:
		 * 
		 * "INVALID_ID"		Invalid Question, Category, or User ID.
		 * "NONEXISTANT"		You are trying to access non existing comments or answers for a Question.
		 * "API_RESPONSE"		The API has responded with a Error message. Usually when search parameters are incorrent.
		 * "XML_LOADING"		The XML file was unable to load or the path does not exist.
		 * "NO_RESULTS"			The API call yielded no results.
		 */
		public function get errorType():String
		{
			return _errorType;
		}
		
		public function set errorType( value:String ):void
		{
			_errorType = value;
		}
	}
	
}