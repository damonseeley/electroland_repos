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
	* Yahoo! Upcoming.org API User Class. This class holds the data for an
	* Upcoming.org user.
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Allen Rabinovich 02/25/07
	*/
	public class User extends EventDispatcher
	{
		/**
		* Defines the parent UpcomingService for this user (used to access the token and API key)
		*/
		public var parent_service:UpcomingService;
		/**
		* The Upcoming.org user's id number.
		*/		
		public var id:Number;
		/**
		* The Upcoming.org user's name.
		*/		
		public var name:String;
		/**
		* The Upcoming.org user's username.
		*/
		public var username:String;
		/**
		* The Upcoming.org user's zip code.
		*/		
		public var zip:String;
		/**
		* The Upcoming.org user's photo URL.
		*/		
		public var photourl:String;
		
		/**
		 * The User class constructor.
		 * @param _parent_service An UpcomingService with an instantiated token and API key.
		 * @param _id The user id.
		 * @param _name The user's name.
		 * @param _username The user's username.
		 * @param _zip The user's zip code.
		 * @param _photourl The user's photo URL.
		 */		
		public function User(_parent_service:UpcomingService, _id:Number, _name:String, _username:String, _zip:String, _photourl:String) {
			parent_service = _parent_service;
			id = _id;
			name = _name;
			username = _username;
			zip = _zip;
			photourl = _photourl;
		}
 		/**
 		 * This method updates the data in the specific instance of User
 		 * by the ID number. If the <code>id</code> property is already filled in,
 		 * then no argument needs to be passed. Otherwise, pass the user's
 		 * Upcoming.org identification number.
 		 * <p>When the update completes, the user class will dispatch an
 		 * <code>UpcomingResultEvent.USER_UPDATED</code> event.</p>
 		 * @param _id The user's Upcoming.org identification number, if
 		 * none is filled in.
 		 * 
 		 */ 		
		public function updateById(_id:Number = NaN):void {
			if (!isNaN(_id)) {
				id = _id;
			}
			sendQuery("user.getInfo", ("user_id=" + id), UpcomingResultEvent.USER_UPDATED, 'GET');
		}
		 /**
 		 * This method updates the data in the specific instance of user
 		 * by the username. If the <code>username</code> property is already filled in,
 		 * then no argument needs to be passed. Otherwise, pass the user's
 		 * Upcoming.org identification number.
 		 * <p>When the update completes, the user class will dispatch an
 		 * <code>UpcomingResultEvent.USER_UPDATED</code> event.</p>
 		 * @param _username The user's Upcoming.org username, if
 		 * none is filled in.
 		 * 
 		 */ 	
		public function updateByUsername(_username:String = null):void {
			if (_username != null) {
				username = _username;
			}
			sendQuery("user.getInfoByUsername", ("username=" + username), UpcomingResultEvent.USER_UPDATED, 'GET');
		}
		/**
		 * This method retrieves a list of Metros for the given user.
		 * When the list if retrieved, the current instance of User
		 * dispatches a <code>UpcomingResultEvent.METRO_GET_LIST</code> event.
		 */	
		public function getMetroList():void {
			var params:String = "token=" + parent_service.token;
			params = params + "&user_id=" + id;
			sendQuery("user.getMetroList", params, UpcomingResultEvent.METRO_GET_LIST, 'GET'); 	
		}
		/**
		 * This retrieves a list of events watched by a specific user.
		 * When the results are received, the current instance of User
		 * dispatches a USER_GET_WATCHLIST event.
		 * 
		 * @param show A string describing which events to show. Can be one of: "upcoming", "past", or "all"
		 */		
		public function getWatchList(show:String = null):void {
			var params:String = "token=" + parent_service.token;
			params = params + "&user_id=" + id;
			
			if (show != null) {
				params = params + "&show=" + show;
			} 
			sendQuery("user.getWatchList", params, UpcomingResultEvent.USER_GET_WATCHLIST, 'GET'); 	
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
 			var variables:URLVariables;
 			if (methodType.toLowerCase() == 'post') {
	 			if(params != null) {
	 				variables = new URLVariables(queryParams);
	            	request.method = URLRequestMethod.POST;
	            	request.data = variables;
	    		}
  			}
  			else if (methodType.toLowerCase() == 'get') {
  				if (params != null) {
  					variables = new URLVariables(queryParams);
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
				var result:Array = new Array();
				
				queryXML = new XML(queryLoader.data);
				
				switch(dispatchType) {
					case UpcomingResultEvent.USER_UPDATED:
						for each (resultNode in queryXML.children()) {
							id = resultNode.@id;
							name = resultNode.@name;
							username = resultNode.@username;
							zip = resultNode.@zip;
							photourl = resultNode.@photourl;
						}
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.USER_UPDATED, this));
						break;
					case UpcomingResultEvent.METRO_GET_LIST:
						for each (resultNode in queryXML.children()) {					
						var mo_mylist:Metro = new Metro(this, resultNode.@id, resultNode.@name, resultNode.@code, new State(this, resultNode.@state_id, resultNode.@state_name, resultNode.@state_code));
						result.push(mo_mylist);
						}			
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.METRO_GET_LIST, result));
						break;
					case UpcomingResultEvent.USER_GET_WATCHLIST:
						for each (resultNode in queryXML.children()) {	
						var new_state:State = new State(this, resultNode.@venue_state_id, resultNode.@venue_state_name, resultNode.@venue_state_code);
						var new_metro:Metro = new Metro(this, resultNode.@metro_id, null, null, new_state);
						var new_venue:Venue = new Venue(this, null, resultNode.@venue_address, resultNode.@venue_city, new_metro, resultNode.@venue_id, resultNode.@venue_zip, resultNode.@venue_phone, resultNode.@venue_url, null, false); 				
						var uE:uEvent = new uEvent(this);
						uE.id = resultNode.@id;
						uE.name = resultNode.@name;
						uE.venue = new_venue;
						uE.description = resultNode.@description;
						uE.start_date = parent_service.dateFromString(resultNode.@start_date, resultNode.@start_time);
						if (resultNode.@end_date != "0000-00-00") {
								uE.end_date = parent_service.dateFromString(resultNode.@end_date, resultNode.@end_date);
						}
						uE.personal = resultNode.@personal==1?true:false;
						uE.selfpromotion = resultNode.@selfpromotion==1?true:false;
						uE.user = new User(this, resultNode.@user_id, null, null, null, null);
						uE.category = new Category(this, resultNode.@category_id, null, null);
						uE.latitude = resultNode.@latitude;
						uE.longitude = resultNode.@longitude;
						uE.geocoding_precision = resultNode.@geocoding_precision;
						uE.geocoding_ambiguous = resultNode.@geocoding_ambiguous==1?true:false;
						result.push({event: uE, status: resultNode.@status});
						}			
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.METRO_GET_LIST, result));
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