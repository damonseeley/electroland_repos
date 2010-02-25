/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.upcoming.events
{
	import flash.events.Event;
	
	/**
	* Error Event class in response to data events from the Yahoo! Upcoming API.
	* 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Scott Morgan 02/25/07
	*/
	public class UpcomingErrorEvent extends Event
	{
		/**
		* The type of event dispatched when the frob is invalid.
		*/		
		public static const INVALID_FROB:String = "InvalidFrob";
		/**
		* The type of event dispatched when XML data could not be loaded from the server.
		*/		
		public static const XML_LOADING:String = "XMLLoading";
		
		/**
		 * True if the event is the result of a successful call,
		 * False if the call failed
		 */
		public var success:Boolean;
		
		private var _data:Object;
		private var _errorType:String;
		
		/**
		 * Constructs a new UpcomingErrorEvent
		 * @param type One of the event types defined by constants
		 * @inData The event data object
		 */
		public function UpcomingErrorEvent(type:String, inData:Object)
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
		
		/**
		 * Error Type
		 * 
		 * Available errorType values:
		 * 
		 * "INVALID_LOCATION"		Invalid location code
		 * "XML_LOADING"		The XML file was unable to load or the path does not exist.
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