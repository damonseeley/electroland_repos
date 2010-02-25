/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.upcoming
{
	import flash.net.URLRequest;
	import flash.net.URLRequestMethod;
	import flash.net.URLLoader;
	import flash.net.URLVariables;

	import flash.events.Event;
	import flash.events.EventDispatcher;
	import com.yahoo.webapis.upcoming.events.*;
	/**
	* Yahoo! Upcoming.org API uEvent Class. This class holds information
	* about individual events in the Upcoming.org event database.
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Allen Rabinovich 02/25/07
	*/
	
public class uEvent extends EventDispatcher
	{
		/**
		* The Upcoming.org event ID
		*/
		public var id:Number;	
		/**
		* Defines the parent UpcomingService for this event (used to access the token and API key)
		*/		
		public var parent_service:UpcomingService;
		/**
		* The event name
		*/		
		public var name:String;
		/**
		* Event description
		*/		
		public var description:String;
		/**
		* The starting date and time for the event
		*/		
		public var start_date:Date;
		/**
		* The ending date and time for the event, if applicable
		*/		
		public var end_date:Date;
		/**
		* The date when the event was posted
		*/		
		public var post_date:Date;
		/**
		* The privacy flag: if <code>true</code>, the event is private; otherwise, public.
		*/		
		public var personal:Boolean;
		/**
		* The selfpromotion flag. If <code>true</code>, then the event is being self-promoted.
		*/
		public var selfpromotion:Boolean;
		/**
		* The metro area to which the event belogs.
		*/		
		public var metro:Metro;
		/**
		* The venue where the event will take place.
		*/		
		public var venue:Venue;
		/**
		* The user announcing the event.
		*/		
		public var user:User;
		/**
		* The event website URL, if applicable.
		*/		
		public var url:String;
		/**
		* The event category (from a preset list of Upcoming.org categories).
		*/		
		public var category:Category;
		/**
		* Latitude of the event's location
		*/		
		public var latitude:Number;
		/**
		* Longitude of the event's location
		*/		
		public var longitude:Number;
		/**
		* String describing the level of precision for the geographic information
		* (for example, "zip", or "address")
		*/		
		public var geocoding_precision:String;
		/**
		* A Boolean indicating whether the geocoding associated with the event is ambiguous.
		*/		
		public var geocoding_ambiguous:Boolean;
		/**
		* List of users watching the event.
		*/		
		public var watchlist:Array;
/**
 * Constructor for the uEvent class. The only required parameter
 * is the _parent_service, which gives uEvent access to the API key and the user token.
 * Other parameters may need to be filled in depending on what methods are called.
 * @param _parent_service An UpcomingService with an instantiated token and API key.
 * @param _name The event name
 * @param _venue The venue where the event is to take place
 * @param _metro The metro area where the event venue is located
 * @param _category The event category, from a preset list of Upcoming.org categories
 * @param _start_date The starting date and time of the event
 * @param _end_date The end date and time for the event
 * @param _description The event description
 * @param _url Event's website URL, if applicable
 * @param _personal Event's privacy setting. If <code>true</code>, the event is private, otherwise it is public.
 * @param _selfpromotion Event's self-promotion flag. If <code>true</code>, then the event is self-promoted.
 */
public function uEvent(_parent_service:UpcomingService, _name:String = null, _venue:Venue = null, _metro:Metro = null, _category:Category = null, _start_date:Date = null, _end_date:Date = null, _description:String = null, _url:String = null, _personal:Boolean = false, _selfpromotion:Boolean = false) {
	parent_service = _parent_service;
	name = _name;
	venue = _venue;
	metro = _metro;
	category = _category;
    start_date = _start_date;
    end_date = _end_date;
    description = _description;
    url = _url;
    personal = _personal;
    selfpromotion = _selfpromotion;	
}

		/**
 		 * This method updates the data in the specific instance of uEvent
 		 * by the ID number. If the <code>id</code> property is already filled in,
 		 * then no argument needs to be passed. Otherwise, pass the uEvent's
 		 * Upcoming.org identification number.
 		 * <p>When the update completes, the uEvent class will dispatch an
 		 * <code>UpcomingResultEvent.EVENT_LOADED</code> event.</p>
 		 * @param _id The uEvent's Upcoming.org identification number, if
 		 * none is filled in.
 		 * 
 		 */ 		
public function update (event_id:Number = NaN) : void {
    if (!isNaN(event_id)) {
    	id = event_id;
    }
    if (parent_service.token != null) {
	sendQuery("event.getInfo", ("event_id=" + id + "&token=" + parent_service.token), UpcomingResultEvent.EVENT_LOADED, 'GET');
	} else {
	sendQuery("event.getInfo", ("event_id=" + id), UpcomingResultEvent.EVENT_LOADED, 'GET');
	}	
}

/**
 * Adds the current event and its filled in data to the Upcoming.org
 * database. The database requires that the name, venue and category
 * are filed in. When the event is added, uEvent dispatched a
 * <code>UpcomingResultEvent.EVENT_ADDED event.
 */
public function add () : void {
			var params:String = "";
			params += "token=" + escape(parent_service.token);
			params += "&name=" + escape(name);
			params += "&venue_id=" + escape(venue.id.toString()); 
			params += "&category_id=" + escape(category.id.toString());
			var month:String = "";
			var day:String = "";
			var hour:String = "";
			var minute:String = "";
			var second:String = "";
			if (start_date.getMonth() < 9) {
				month = "0";
			}
			if (start_date.getDate() < 10) {
				day = "0";
			}
			if (start_date.getHours() < 10) {
				hour = "0";
			}
			if (start_date.getMinutes() < 10) {
				minute = "0";
			}
			if (start_date.getSeconds() < 10) {
				second = "0";
			}
			params += "&start_date=" + start_date.getFullYear().toString() + "-" + month + (start_date.getMonth()+1) + "-" + day + start_date.getDate().toString(); 
			params += "&start_time=" + hour + start_date.getHours() + ":" + minute + start_date.getMinutes().toString() + ":" + second + start_date.getSeconds().toString();
			if (end_date != null) {
			month = "";
			day = "";
			hour = "";
			minute = "";
			second = "";
			if (end_date.getMonth() < 9) {
				month = "0";
			}
			if (end_date.getDate() < 10) {
				day = "0";
			}
			if (end_date.getHours() < 10) {
				hour = "0";
			}
			if (end_date.getMinutes() < 10) {
				minute = "0";
			}
			if (end_date.getSeconds() < 10) {
				second = "0";
			}
				params += "&end_date=" + end_date.getFullYear() + "-" + month + (end_date.getMonth()+1) + "-" + day + end_date.getDate(); 
				params += "&end_time=" + end_date.getHours() + ":" + minute + end_date.getMinutes() + ":" + second + end_date.getSeconds();
			}
			
			if (url != null && url != '') {
				params += "&url=" + escape(url);
			}
			
			if (description != null && description != '') {
				params += "&description=" + escape(description);
			}
			
			if (personal) {
				params += "&personal=1";
			} else {params += "&personal=0";}
			
			if (selfpromotion) {
				params += "&selfpromotion=1";} else {params += "&selfpromotion=0"}
			
			sendQuery("event.add", (params), UpcomingResultEvent.EVENT_ADDED);
}

/**
 * Updates an existing event with the current event's instance's data fields.
 * Requires that the event's ID is filled in; if it's not, it adds the event
 * as a new event instead.
 * When editing an event is complete, the uEvent instance dispatches 
 * <code>UpcomingResultEvent.EVENT_EDITED</code> event.
 * 
 */
public function commit () : void {
	if (isNaN(id)) {
		add();
	}
	else {
			var params:String = "";
			params += "token=" + escape(parent_service.token);
			params += "&event_id=" + id;
			params += "&name=" + escape(name);
			params += "&venue_id=" + escape(venue.id.toString()); 
			params += "&category_id=" + escape(category.id.toString());
			params += "&metro_id=" + metro.id; 
			
			var month:String = "";
			var day:String = "";
			var hour:String = "";
			var minute:String = "";
			var second:String = "";
			if (start_date.getMonth() < 9) {
				month = "0";
			}
			if (start_date.getDate() < 10) {
				day = "0";
			}
			if (start_date.getHours() < 10) {
				hour = "0";
			}
			if (start_date.getMinutes() < 109) {
				minute = "0";
			}
			if (start_date.getSeconds() < 10) {
				second = "0";
			}
			params += "&start_date=" + start_date.getFullYear() + "-" + month + (start_date.getMonth()+1) + "-" + day + start_date.getDate(); 
			params += "&start_time=" + hour + start_date.getHours() + ":" + minute + start_date.getMinutes() + ":" + second + start_date.getSeconds();
			
			if (end_date != null) {
			month = "";
			day = "";
			hour = "";
			minute = "";
			second = "";
			if (end_date.getMonth() < 9) {
				month = "0";
			}
			if (end_date.getDate() < 10) {
				day = "0";
			}
			if (end_date.getHours() < 10) {
				hour = "0";
			}
			if (end_date.getMinutes() < 10) {
				minute = "0";
			}
			if (end_date.getSeconds() < 10) {
				second = "0";
			}
				params += "&end_date=" + end_date.getFullYear() + "-" + month + (end_date.getMonth()+1) + "-" + day + end_date.getDate(); 
				params += "&end_time=" + end_date.getHours() + ":" + minute + end_date.getMinutes() + ":" + second + end_date.getSeconds();
			}
			if (url != null && url != '') {
				params += "&url=" + escape(url);
			}
			
			if (description != null && description != '') {
				params += "&description=" + escape(description);
			}
			
			if (personal) {
				params += "&personal=1";
			} else {params += "&personal=0";}
			
			if (selfpromotion) {
				params += "&selfpromotion=1";} else {params += "&selfpromotion=0"}
			
			sendQuery("event.edit", (params), UpcomingResultEvent.EVENT_EDITED);
	}
}


/**
 * Gets a list of users currently watching or attending
 * this event. The event id is required to be filled in. 
 * When the list of users is received, uEvent dispatches
 * <code>UpcomingResultEvent.USERS_WATCHING_EVENT</code>, 
 * with an array of Users contained in the event object's 
 * <code>data</code> field.
 */
public function getUsersWatching():void {
    if (parent_service.token != null) {
	sendQuery("event.getWatchlist", ("event_id=" + id + "&token=" + parent_service.token), UpcomingResultEvent.USERS_WATCHING_EVENT, 'GET');
	} else {
	sendQuery("event.getWatchlist", ("event_id=" + id), UpcomingResultEvent.USERS_WATCHING_EVENT, 'GET');
	}	
		}


		/**
		 * This method is used to send the appropriate REST query and parse
		 * the resulting XML.
		 * @private
		 * @param method Upcoming.org API method that should be called.
		 * @param params Parameters to pass in the REST query.
		 * @param dispatchType Event to dispatch when results are received.
		 * @param methodType Whether to POST or to GET the API request.
		 * 
		 */		
		private function sendQuery(method:String, params:String, dispatchType:String, methodType:String = 'POST') : void {
			
		var sendQuery:String;
			sendQuery = (parent_service.api_URL);
		var queryParams:String;
		
			if(params != null) {
				queryParams = ("api_key=" + parent_service.api_id + "&method=" + method + "&"+params);
			} else {
				queryParams = ("api_key=" + parent_service.api_id + "&method=" + method);
			}
			
			var queryXML:XML;
 			var request:URLRequest = new URLRequest(sendQuery);
 			var variables:URLVariables = new URLVariables(queryParams);
 			
 			if (methodType.toLowerCase() == 'post') {
	 			if(params != null) {
	            	request.method = URLRequestMethod.POST;
	           	request.data = variables;
	    		}
  			}
			else if (methodType.toLowerCase() == 'get') {
  				if (params != null) {
 	            	request.method = URLRequestMethod.GET;
	            	request.data = variables;
  				}
  			}

			var queryLoader:URLLoader = new URLLoader(request);
			queryLoader.addEventListener("complete", xmlLoaded);
			queryLoader.addEventListener("ioError", xmlError);	

			
			// XML loading result method
			function xmlLoaded(evtObj:Object):void {
				
				var data:Object = new Object();
				var resultNode:XML;
				var resultObject:Object = new Object();
				
				queryXML = new XML(queryLoader.data);
				switch(dispatchType) {
				case UpcomingResultEvent.USERS_WATCHING_EVENT:
				var result:Array = new Array();
				for each (resultNode in queryXML.children()) {
					var cur_user:User = new User(parent_service, resultNode.@id, resultNode.@name, resultNode.@username, resultNode.@zip, resultNode.@photourl);
					result.push({user: cur_user, status:resultNode.@status});
				}
				dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.USERS_WATCHING_EVENT, result)); 
				break;
				case UpcomingResultEvent.EVENT_LOADED:
				for each (resultNode in queryXML.children()) {
							id = resultNode.@id;
							name = resultNode.@name;
							description = resultNode.@description;
                            start_date = new Date();
                            var sdate:Array = resultNode.@start_date.split("-");
                            var stime:Array = resultNode.@start_time.split(":");
                            start_date.setFullYear(Number(sdate[0]), Number(sdate[1]), Number(sdate[2]));
                            start_date.setHours(Number(stime[0]), Number(sdate[1]), Number(sdate[2]));
                            if (resultNode.@end_date != "0000-00-00") {
                            var edate:Array = resultNode.@end_date.split("-");
                            var etime:Array = resultNode.@end_time.split(":");
                            end_date = new Date();
                            end_date.setFullYear(Number(edate[0]), Number(edate[1])-1, Number(edate[2]));
                            end_date.setHours(Number(etime[0]), Number(edate[1]), Number(edate[2])); 	
                            }
                           
							category = new Category(parent_service, resultNode.@category_id, null, null);
							metro = new Metro(parent_service, resultNode.@metro_id, null, null, null);
							venue = new Venue(parent_service, null, null, null, metro, resultNode.@venue_id);
							user = new User(parent_service, resultNode.@user_id, null, null, null, null);
							
			                var pdate:Array = resultNode.@date_posted.split("-");
			                post_date = new Date();
                            post_date.setFullYear(Number(pdate[0]), Number(pdate[1])-1, Number(pdate[2]));
						}
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.EVENT_LOADED, this)); 
						break;
						
					case UpcomingResultEvent.EVENT_ADDED:
						for each (resultNode in queryXML.children()) {
							id = resultNode.@id;
							name = resultNode.@name;
							description = resultNode.@description;
                            start_date = parent_service.dateFromString(resultNode.@start_date, resultNode.@start_time);
                            if (resultNode.@end_date != "0000-00-00") {
							end_date = parent_service.dateFromString(resultNode.@end_date, resultNode.@end_time);
                            }
							category = new Category(parent_service, resultNode.@category_id, null, null);
							metro = new Metro(parent_service, resultNode.@metro_id, null, null, null);
							venue = new Venue(parent_service, null, null, null, metro, resultNode.@venue_id);
							user = new User(parent_service, resultNode.@user_id, null, null, null, null);
			                post_date = parent_service.dateFromString(resultNode.@date_posted, "00:00:00");
						}
						
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.EVENT_ADDED, this)); 
						break;
						
					case UpcomingResultEvent.EVENT_EDITED:
						for each (resultNode in queryXML.children()) {
							id = resultNode.@id;
							name = resultNode.@name;
							description = resultNode.@description;
                            start_date = parent_service.dateFromString(resultNode.@start_date, resultNode.@start_time);
                            
                            if (resultNode.@end_date != "0000-00-00") {
							end_date = parent_service.dateFromString(resultNode.@end_date, resultNode.@end_time);
                            }                 
							category = new Category(parent_service, resultNode.@category_id, null, null);
							metro = new Metro(parent_service, resultNode.@metro_id, null, null, null);
							venue = new Venue(parent_service, null, null, null, metro, resultNode.@venue_id);
							user = new User(parent_service, resultNode.@user_id, null, null, null, null);
							post_date = parent_service.dateFromString(resultNode.@date_posted, "00:00:00");
 							}
						
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.EVENT_EDITED, this)); 
						break;
 					}
				}
				
			
			function xmlError (evtObj:Object):void {
	        	var error:UpcomingErrorEvent = new UpcomingErrorEvent(UpcomingErrorEvent.XML_LOADING, evtObj);
				dispatchEvent(error);
	        }
		}





	}

}