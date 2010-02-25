/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.weather.events
{
	import flash.events.Event;
	
	/**
	* Event class in response to data events from the Yahoo! Weather API.
	* 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Allen Rabinovich 02/10/07
	*/
	public class WeatherResultEvent extends Event
	{
		/**
		 * True if the event is the result of a successful call,
		 * False if the call failed
		 */
		public var success:Boolean;
		private var _data:Object;
			/**
		* The type of event dispatched when Weather data has loaded successfully.
		*/	
		public static const WEATHER_LOADED:String = "weatherLoaded";
		/**
		 * Constructs a new WeatherResultEvent
		 */
		public function WeatherResultEvent(type:String, inData:Object)
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