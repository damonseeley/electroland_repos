/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.upcoming.events
{
	import flash.events.Event;
	
	/**
	* Event class in response to data events from the Yahoo! Upcoming API.
	* 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Allen Rabinovich 02/25/07
	*/
	public class UpcomingResultEvent extends Event
	{
		/**
		* The type of event dispatched when a group is updated with data from the server.
		*/		
		public static const GROUP_LOADED:String = "GroupLoaded";
		/**
		* The type of event dispatched when the group is added to the server.
		*/		
		public static const GROUP_ADDED:String = "GroupAdded";
		/**
		* The type of event dispatched when the existing group is edited and changes
		 * are commited to the server.
		*/		
		public static const GROUP_EDITED:String = "GroupEdited";
		/**
		* The type of event dispatched when a metro is updated with data from the server.
		*/				
		public static const METRO_GET_INFO:String = "MetroGetInfo";
		/**
		* The type of event dispatched when the list of metros is retrieved.
		*/		
		public static const METRO_GET_LIST:String = "MetroGetList";
		/**
		* The type of event dispatched when the current user's list of metros is retrieved.
		*/		
		public static const METRO_GET_MY_LIST:String = "MetroGetMyList";
		
		/**
		* The type of event dispatched when the list of states is retrieved.
		*/		
		public static const GET_STATE_LIST:String = "MetroGetStateList";
		/**
		* The type of event dispatched when the list of countries is retrieved.
		*/		
		public static const GET_COUNTRY_LIST:String = "MetroGetCountryList";
		/**
		* The type of event dispatched when a metro search is complete.
		*/		
		public static const METRO_SEARCH:String = "MetroSearch";
		/**
		* The type of event dispatched when a metro closest to a given lat/lon pair is found.
		*/		
		public static const METRO_GET_LAT_LON:String = "MetroGetLatLon";
		/**
		* The type of event dispatched when a state is updated with data from the server.
		*/				
		public static const STATE_GET_INFO:String = "StateGetInfo";
		/**
		* The type of event dispatched when a country is updated with data from the server.
		*/				
		public static const COUNTRY_GET_INFO:String = "CountryGetInfo";
		/**
		* The type of event dispatched when a list of all event categories is retrieved.
		*/				
		public static const CATEGORY_GET_LIST:String = "CategoryGetList";
		/**
		* The type of event dispatched when a venue is updated with data from the server.
		*/				
		public static const VENUE_LOADED:String = "VenueGetInfo";
		/**
		* The type of event dispatched when a list of venues is retrieve.
		*/				
		public static const VENUE_GET_LIST:String = "VenueGetList";
		/**
		* The type of event dispatched when a search for venues is complete and results are retrieved.
		*/				
		public static const VENUE_SEARCH:String = "VenueSearch";
		/**
		* The type of event dispatched when a new venue is added to the server.
		*/				
		public static const VENUE_ADDED:String = "VenueAdded";
		/**
		* The type of event dispatched when an existing venue is edited.
		*/				
		public static const VENUE_EDITED:String = "VenueEdited";
		/**
		* The type of event dispatched when an event search is complete and results are retrieved.
		*/						
		public static const EVENT_SEARCH:String = "FindEvents";
		/**
		* The type of event dispatched when an existing event is edited and changes are commited to the server.
		*/				
		public static const EVENT_EDITED:String = "EventEdited";
		/**
		* The type of event dispatched when an event is updated with data from the server.
		*/				
		public static const EVENT_LOADED:String = "EventLoaded";
		/**
		* The type of event dispatched when new event is added to the server.
		*/				
		public static const EVENT_ADDED:String = "EventAdded";
		/**
		* The type of event dispatched when a list of users watching an event is retrieved.
		*/				
		public static const USERS_WATCHING_EVENT:String = "UsersWatchingEvent";
		/**
		* The type of event dispatched when a user is updated with data from the server.
		*/				
		public static const USER_UPDATED:String = "UserUpdated";
		/**
		* The type of event dispatched when a watchlist for a particular user is retrieved.
		*/				
		public static const USER_GET_WATCHLIST:String = "UserGetWatchlist";
		/**
		* The type of event dispatched when a watchlist is retrieved.
		*/				
		public static const WATCHLIST_GET_LIST:String = "WatchlistGetList";
		/**
		* The type of event dispatched when an event is added to user's watchlist.
		*/				
		public static const WATCHLIST_ADDED:String = "WatchlistAdded";
		/**
		* The type of event dispatched when an event is removed from the user's watchlist.
		*/				
		public static const WATCHLIST_REMOVED:String = "WatchlistRemoved";
		/**
		* The type of event dispatched when a user authentication token is obtained.
		*/				
		public static const GET_TOKEN:String = "GetToken";
		
		/**
		 * True if the event is the result of a successful call,
		 * False if the call failed
		 */
		public var success:Boolean;
		
		private var _data:Object;
		
		/**
		 * Constructs a new UpcomingResultEvent
		 * @param type The type of the event defined by one of the constants in this class
		 * @param inData the data object of the event.
		 */
		public function UpcomingResultEvent(type:String, inData:Object)
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