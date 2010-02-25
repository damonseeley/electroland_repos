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
	* Yahoo! Upcoming.org API Venue Class. This class holds the data about
	* Upcoming.org event venues.
	* 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Scott Morgan and Allen Rabinovich 02/25/07
	*/
	public class Venue extends EventDispatcher
	{
		/**
		* Defines the parent UpcomingService for this country (used to access the token and API key)
		*/
		public var parent_service:UpcomingService;
		/**
		* The Upcoming.org venue id number.
		*/	
		public var id:Number;
		/**
		* The venue name.
		*/		
		public var name:String;
		/**
		* The venue street address.
		*/		
		public var address:String;
		/**
		* The venue city.
		*/		
		public var city:String;
		/**
		* The venue zip.
		*/		
		public var zip:String;
		/**
		* The venue phone number.
		*/		
		public var phone:String;
		/**
		* The venue web site URL.
		*/
		public var url:String;
		/**
		* The venue description
		*/		
		public var description:String;
		/**
		* The metro where the venue is located.
		*/		
		public var metro:Metro;
		/**
		* The user associated with the venue (creator).
		*/		
		public var user:User;
		/**
		* The venue privacy setting: if <code>true</code>, then the venue is private.
		*/		
		public var privacy:Boolean;
		/**
		* The venue latitude.
		*/		
		public var latitude:Number;
		/**
		* The venue longitude.
		*/		
		public var longitude:Number;
		/**
		* The string describing the geocoding precision for the venue, for example "address" or "zip".
		*/		
		public var geocoding_precision:String;
		/**
		* The geocoding ambiguity flag; if <code>true</code>, then the geocoding is ambiguous.
		*/		
		public var geocoding_ambiguous:Boolean;

 		/**
 		 * Venue class constructor.
 		 * @param _parent_service An UpcomingService with an instantiated token and API key.
 		 * @param _name The venue name.
 		 * @param _address The venue address.
 		 * @param _city The venue city.
 		 * @param _metro The venue metro.
 		 * @param _id The venue Upcoming.org id number.
 		 * @param _zip The venue zip code.
 		 * @param _phone The venue phone number.
 		 * @param _url The venue website URL.
 		 * @param _description The description of the venue.
 		 * @param _private The venue privacy setting. If <code>true</code>, the venue is private.
 		 */		
		
 		public function Venue (_parent_service:UpcomingService, _name:String = null, _address:String = null, _city:String = null, _metro:Metro = null, _id:Number = NaN, _zip:String = null, _phone:String = null, _url:String = null, _description:String = null, _private:Boolean = false) {
 		parent_service = _parent_service;
 		id = _id;
 		name = _name;
 		address = _address;
 		city = _city;
 		metro = _metro;
 		zip = _zip;
 		phone = _phone;
 		url = _url;
 		description = _description;
 		privacy = _private;
 		}
		
 		/**
 		 * This method updates the data in the specific instance of Venue
 		 * by the ID number. If the <code>id</code> property is already filled in,
 		 * then no argument needs to be passed. Otherwise, pass the venue's
 		 * Upcoming.org identification number.
 		 * <p>When the update completes, the venue class will dispatch an
 		 * <code>UpcomingResultEvent.VENUE_LOADED</code> event.</p>
 		 * @param _id The venue's Upcoming.org identification number, if
 		 * none is filled in.
 		 * 
 		 */ 			
		public function update(_id:Number = NaN):void {
			if (!isNaN(_id)) {
			id = _id;
			}
		if (parent_service.token != null) {
				sendQuery("venue.getInfo", ("venue_id=" + id + "&token=" + parent_service.token), UpcomingResultEvent.VENUE_LOADED, 'GET');
			} else {
				sendQuery("venue.getInfo", ("venue_id=" + id), UpcomingResultEvent.VENUE_LOADED, 'GET');
			}			
		}
		/**
 * Adds the current venue and its filled in data to the Upcoming.org
 * database. The database requires that the name, address, city and metro
 * are filled in. When the venue is added, the current instance of Venue dispatches a
 * <code>UpcomingResultEvent.VENUE_ADDED event.
 */
		public function add():void {
			var params:String = "";
			params += "token=" + escape(parent_service.token);
			params += "&venuename=" + escape(name);
			params += "&venueaddress=" + escape(address); 
			params += "&venuecity=" + escape(city);
			params += "&metro_id=" + metro.id; 
			if (privacy) {
				params += "&private=1";
			}
			else {
				params += "&private=0";
			}
			if (zip != null && zip != '') {
				params += "&venuezip=" + escape(zip);
			}
			if (phone != null && phone != '') {
				params += "&venuephone=" + escape(phone);
			}
			if (url != null && url != '') {
				params += "&venueurl=" + escape(url);
			}
			if (description != null && description != '') {
				params += "&venuedescription=" + escape(description);
			}
			
			sendQuery("venue.add", (params), UpcomingResultEvent.VENUE_ADDED);
		}
		
/**
 * Updates an existing venue with the current venue's instance's data fields.
 * Requires that the venue's ID is filled in; if it's not, it adds the venue
 * as a new venue instead.
 * When editing a venue is complete, the Venue instance dispatches 
 * <code>UpcomingResultEvent.VENUE_EDITED</code> event.
 * 
 */
		public function commit():void {
			
			if (isNaN(id)) {
				add();
			}
			else {
			var params:String = "";
			params += "token=" + escape(parent_service.token);
			params += "&venuename=" + escape(name);
			params += "&venueaddress=" + escape(address); 
			params += "&venuecity=" + escape(city);
			params += "&metro_id=" + metro.id;
			params += "&venue_id=" + id; 
			if (privacy) {
				params += "&private=1";
			} 
			else {
				params += "&private=0";
			}
			if (zip != null && zip != '') {
				params += "&venuezip=" + escape(zip);
			}
			if (phone != null && phone != '') {
				params += "&venuephone=" + escape(phone);
			}
			if (url != null && url != '') {
				params += "&venueurl=" + escape(url);
			}
			if (description != null && description != '') {
				params += "&venuedescription=" + escape(description);
			}			
			sendQuery("venue.edit", (params), UpcomingResultEvent.VENUE_EDITED);
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
					case UpcomingResultEvent.VENUE_LOADED:
						for each (resultNode in queryXML.children()) {
							id = resultNode.@id;
							name = resultNode.@name;
							address = resultNode.@address;
							city = resultNode.@city;
							zip = resultNode.@zip;
							phone = resultNode.@phone;
							url = resultNode.@url;
							description = resultNode.@description;
							user = new User(parent_service, resultNode.@user_id, null, null, null, null);
							privacy = new Boolean(resultNode.@private);
							latitude = resultNode.@latitude;
							longitude = resultNode.@longitude;
							geocoding_precision = resultNode.@geocoding_precision;
							geocoding_ambiguous = resultNode.@geocoding_ambiguous==0?false:true;
							data = this;
						}
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.VENUE_LOADED, this)); 
						break;
					case UpcomingResultEvent.VENUE_GET_LIST:
						for each (resultNode in queryXML.children()) {
								resultObject.id = resultNode.@id;
								resultObject.name = resultNode.@name;
								resultObject.city = resultNode.@city;
								resultObject.url = resultNode.@url;
								resultObject.user_id = resultNode.@user_id;
								resultObject.private = resultNode.@private;
								resultObject.latitude = resultNode.@latitude;
								resultObject.longitude = resultNode.@longitude;
								resultObject.geocoding_precision = resultNode.@geocoding_precision;
								resultObject.geocoding_ambiguous = resultNode.@geocoding_ambiguous;
								data.push(resultObject);							
						}

					case UpcomingResultEvent.VENUE_ADDED:
						for each (resultNode in queryXML.children()) {
							id = resultNode.@id;
							name = resultNode.@name;
							address = resultNode.@address;
							city = resultNode.@city;
							zip = resultNode.@zip;
							phone = resultNode.@phone;
							url = resultNode.@url;
							description = resultNode.@description;
							user = new User(parent_service, resultNode.@user_id, null, null, null, null);
							privacy = new Boolean(resultNode.@private);
							latitude = resultNode.@latitude;
							longitude = resultNode.@longitude;
							geocoding_precision = resultNode.@geocoding_precision;
							geocoding_ambiguous = resultNode.@geocoding_ambiguous==0?false:true;
							data = this;
						}
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.VENUE_ADDED, this)); 
						break;
					case UpcomingResultEvent.VENUE_EDITED:
						for each (resultNode in queryXML.children()) {
							id = resultNode.@id;
							name = resultNode.@name;
							address = resultNode.@address;
							city = resultNode.@city;
							zip = resultNode.@zip;
							phone = resultNode.@phone;
							url = resultNode.@url;
							description = resultNode.@description;
							user = new User(parent_service, resultNode.@user_id, null, null, null, null);
							privacy = new Boolean(resultNode.@private);
							latitude = resultNode.@latitude;
							longitude = resultNode.@longitude;
							geocoding_precision = resultNode.@geocoding_precision;
							geocoding_ambiguous = resultNode.@geocoding_ambiguous==0?false:true;
							data = this;
						}
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.VENUE_EDITED, this)); 
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