/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.search.events
{
	import flash.events.Event;
	import com.yahoo.webapis.ServiceFault;
	
	/**
	* Error Event class in response to fault events from the Yahoo! Search API.
	* 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Alaric Cole 02/22/07
	*/
	public class SearchFaultEvent extends Event
	{
		/** Constant for the event types. */
		public static const FAULT:String = "fault";
		
		public static const ERROR_EVENT:String = "ErrorEvent";
		
		public static const API_RESPONSE:String  = "API Response";
		public static const XML_LOADING:String  = "XML Loading";
		public static const NO_RESULTS:String  = "No Results";
				
		/**
		 * True if the event is the result of a successful call,
		 * False if the call failed
		 */
		public var success:Boolean;
		
		private var _fault:ServiceFault;
		
		/**
		 * Constructs a new ErrorEvent
		 */
		public function SearchFaultEvent(type:String, fault:ServiceFault = null)
		{
			if(fault!= null) this._fault = fault;
									   	
			super( type, bubbles, cancelable );
		}
		
		
		/**
		 * Fault
		 */
		public function get fault():ServiceFault
		{
			return _fault;
		}
		
		public function set fault( value:ServiceFault ):void
		{
			_fault = value;
		}
		
		
		
	}
	
}