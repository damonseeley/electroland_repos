/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.upcoming
{
	import flash.net.URLRequest;
	import flash.net.URLLoader;
	import flash.events.Event;
	import flash.events.EventDispatcher;
	import flash.net.URLRequestMethod;
	import flash.net.URLVariables;
	
	import com.yahoo.webapis.upcoming.events.*;
	
	/**
	* Yahoo! Upcoming.org API Service Class. This class is the starting point for the Upcoming.org
	* AS3 Library. It provides the API and user authentication mechanisms, and a number of functions
	* for retrieving Upcoming.org data.
	* 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Allen Rabinovich 02/25/07
	*/
	public class UpcomingService extends EventDispatcher
	{
		
 		private var _api_id:String;
 		public var _user:User;

 		private var _api_URL:String = "http://upcoming.yahooapis.com/services/rest/";
 		private var _auth_URL:String = "http://www.upcoming.org/services/auth/?";
 		private var _token:String;
 				
 		private var uResults:UpcomingResultEvent;
 		private var uError:UpcomingErrorEvent;
 		
 		public var _self:UpcomingService;
		
		public const tokenEvent:String = "TokenEvent";

		/**
		 * UpcomingService constructor.
		 * @param apikey The API key for your application. To register a new API key,
		 * follow the instructions provided <a href="http://upcoming.org" target="_blank">here</a>
		 */		
		public function UpcomingService( apikey:String ) {
			_api_id = apikey;
			_self = this;
		}
		/**
		 * The user currently authenticated via this instance of UpcomingService.
		 * 
		 */
		public function get user () : User {
			return _user;
		}
		
		public function set user(value:User) : void {
			_user = value;
		}
	
		/**
		 * The API key currently associated with this instance of UpcomingService.
		 * 
		 */
		public function get api_id () : String {
			return _api_id;
		}
		
		public function set api_id (value:String) : void {
			_api_id = api_id;
		}

		/**
		 * The API URL currently associated with this instance of UpcomingService.
		 * The default value is set to the current API entrypoint, but can be changed if necessary.
		 */	
		public function get api_URL ():String {
			return _api_URL;
		}
		
		public function set api_URL (value:String):void {
			_api_URL = value;
		}
	    /**
	     * The user token currently associated with this instance of UpcomingService.
	     * @see getToken
	     */	
	    public function get token ():String {
	    	return _token;
	    }
	    
	    public function set token(value:String):void {
	    	_token = value;
	    }
	    
		/**
		 * Using the frob received from the user or from Upcoming.org via
		 * a callback, this function retrieves and sets the <code>token</code>
		 * property. When the token is received, this instance of UpcomingService
		 * dispatches an <code>UpcomingResultEvent.GET_TOKEN</code> event.
		 * If an error occurs, the event dispatched is instead a <code>UpcomingErrorEvent.INVALID_FROB</code>,
		 * or an <code>UpcomingErrorEvent.XML_LOADING</code>.
		 * @param frob A code obtained by the user from the authentication URL or passed to the application
		 * via a URL callback.
		 * */
		public function getToken(frob:String) : void {
			var queryXML:XML;
			var queryURL:String = _api_URL + "?method=auth.getToken" + "&api_key=" + _api_id + "&frob=" + frob;
 			var queryXMLURL:URLRequest = new URLRequest(queryURL);
			var queryLoader:URLLoader = new URLLoader(queryXMLURL);
			queryLoader.addEventListener("complete", tokenLoaded);
			queryLoader.addEventListener("ioError", tokenError);
			
			function tokenLoaded(evtObj:Object):void
			{
			queryXML = new XML(queryLoader.data);
			if (queryXML.error.@msg == "Frob not found or was invalid.") {
			uError = new UpcomingErrorEvent(UpcomingErrorEvent.INVALID_FROB, "The frob is invalid");
			dispatchEvent(uError);
			}
			else {
			token = queryXML.token.@token;
			user = new User(_self, queryXML.token.@user_id, queryXML.token.@user_name, queryXML.token.@user_username, null, null);
			uResults = new UpcomingResultEvent(UpcomingResultEvent.GET_TOKEN, _token);
			dispatchEvent(uResults);
			}
     		}
     		
     		function tokenError(evtObj:Object):void
  			{
			uError = new UpcomingErrorEvent(UpcomingErrorEvent.XML_LOADING, "XML could not be loaded");
			dispatchEvent(uError);
  			}
		}	

		/**
		 * This method returns the authentication URL where the user can
		 * approve the application's access to his or her private data. Once approved,
		 * the user will either be issued a frob to copy/paste back into your application,
		 * or, if you have supplied a callback URL, the user will be redirected to that URL
		 * with the frob passed as a parameter.
		 * @return The authentication URL for your application.
		 * 
		 */		
		public function getFrobURL():String {
		   return _auth_URL + "api_key=" + _api_id;
		}

		/**
		 * If the token is valid, this method will return <code>true</code>. Otherwise, it'll return <code>false</code>
		 * @return A Boolean value indicating whether the token is valid.
		 * 
		 */
		public function checkToken() : Boolean	{
			if (token != null) 
				return true;
			else
				return false;
		}		
		
		/**
		 * This method allows you to search for events given a set of parameters. When the events are found,
		 * the current instance of UpcomingService dispatches a <code>UpcomingResultEvent.EVENT_SEARCH</code>event.
		 * None of the parameters are required, and any combination of parameters can be included.
		 * 
		 * @param search_text Keywords to look for in event names and descriptions
		 * @param location The event location in the form similar to "City, State" or a comma-separated pair of latitude
		 * and longitude.
		 * @param radius Radius, in miles, to search arond the specified location.
		 * @param country Country where the event takes place.
		 * @param state State where the event takes place.
		 * @param metro Metro area where the event takes place
		 * @param venue A particular venue where the event takes place. 
		 * @param category An event category, from a predefined Upcoming.org category list.
		 * @param min_date The minimum of the date range to search in.
		 * @param max_date The maximum of the date range to search in.
		 * @param tags Comma-separated event tags.
		 * @param per_page Number of results returned per page.
		 * @param page Number of the page of results.
		 * @param sort Sorting order for the results. Can be one of the following: name-desc, name-asc, start-date-desc, start-date-asc, posted-date-asc, posted-date-desc, distance-asc, distance-desc.
		 * 
		 */
		public function findEvents (search_text:String = null , location:String = null, radius:Number = NaN, country:Country = null, state:State = null, metro:Metro = null, venue:Venue = null, category:Category = null, min_date:Date = null, max_date:Date = null, tags:Array = null, per_page:Number = NaN, page:Number = NaN, sort:String = null) : void {
			var params:String = "";
			
			if (token != null) {
			params += "token=" + escape(token);
			}
			
			if (search_text != null && search_text != '') {
				params += "&search_text=" + escape(search_text);			
			}
			
			if (location != null && location != '') {
				params += "&location=" + escape(location);
			}
			
			if (!isNaN(radius)) {
				params += "&radius=" + radius;
			}
			
			if (country != null) {
				params += "&country_id=" + country.id;
			} 
			
			if (state != null) {
				params += "&state_id=" + state.id;
			}
			
			if (metro != null) {
				params += "&metro_id=" + metro.id;
			}
				
			if (venue != null) {
				params += "&venue_id=" + venue.id;
			}
			
			if (category != null) {
				params += "&category_id=" + category.id;
			}
			
			if (min_date != null) {
				params += "&min_date=" + stringFromDate(min_date);
			}
			
			if (max_date != null) {
				params += "&max_date=" + stringFromDate(max_date);
			}
			
			if (tags != null) {
				params += "&tags=" + tags.toString(); 
			}
			
			if (!isNaN(per_page)) {
				params += "&per_page=" + per_page;
			}
			
			if (!isNaN(page)) {
				params += "&page=" + page;
			}
		
		    if (sort != null) {
		    	params += "&sort=" + sort;
		    }
			sendQuery("event.search", (params), UpcomingResultEvent.EVENT_SEARCH, 'GET');
		}
		
		/**
		 * @private
		 * Metho for converting Upcoming.org date and time formats into a single date
		 * @param date
		 * @param time
		 * @return 
		 * 
		 */
		public function dateFromString(date:String, time:String): Date {
		var new_date:Date = new Date();
        var sdate:Array = date.split("-");
        var stime:Array = time.split(":");
        if (date != "") {
        new_date.setFullYear(Number(sdate[0]), Number(sdate[1])-1, Number(sdate[2]));
        }
        if (time != "") {
        new_date.setHours(Number(stime[0]), Number(stime[1]), Number(stime[2]));
        }
        return new_date;
		}
		
		/**
		 * @private
		 * Method for converting a date into a date string.
		 * @param date
		 * @return 
		 * 
		 */
		public function stringFromDate(date:Date):String {

			var month:String = "";
			var day:String = "";

			if (date.getMonth() < 9) {
				month = "0";
			}
			if (date.getDate() < 10) {
				day = "0";
			}
				return String(date.getFullYear()) + "-" + month + String(date.getMonth()+1) + "-" + day + String(date.getDate()); 
			}

		/**
		 * @private
		 * Method for converting a date into a time string.
		 * @param date
		 * @return 
		 * 
		 */
		public function stringFromTime(date:Date):String {
			var hour:String = "";
			var minute:String = "";
			var second:String = "";
	
			if (date.getHours() < 10) {
				hour = "0";
			}
			if (date.getMinutes() < 10) {
				minute = "0";
			}
			if (date.getSeconds() < 10) {
				second = "0";
			}
			return String(date.getHours()) + ":" + minute + String(date.getMinutes()) + ":" + second + String(date.getSeconds());
		}

		/**
		 * This method allows you to find a list of metros based on specific parameters.
		 * Once the metros have been found, the current instance of UpcomingService dispatches
		 * an event UpcomingResultEvent.METRO_SEARCH.
		 * 
		 * @param search_text Keywords to look for in the name or description of the metro.
		 * @param country Country where the metro area is located.
		 * @param state State where the metro area is located.
		 * 
		 */
		public function findMetros (search_text:String = null, country:Country = null, state:State = null) : void {
			var params:String = "";
			
			if (search_text != null && search_text != '') {
				params += "&search_text=" + escape(search_text);			
			}
			
			if (country != null) {
				params += "&country_id=" + country.id;
			}
			
			if (state != null) {
				params += "&state_id=" + state.id;
			}

			sendQuery("metro.search", (params), UpcomingResultEvent.METRO_SEARCH, 'GET');
		}

		/**
		 * This method allows you to find venues where events take place, based on specific 
		 * parameters. When the list of events is found, the current instance of UpcomingService dispatches a
		 * <code>UpcomingResultEvent.VENUE_SEARCH</code>event. 
		 * @param search_text Keywords to look for in the venue name or description.
		 * @param location Location of the venue, in the form similar to "City, state" or "zip".
		 * @param radius Radius in miles in which to search around the location.
		 * @param country Country where the venue is located.
		 * @param state State where the venue is located.
		 * @param metro Metro area where the venue is located
		 * @param per_page Number of results returned per page
		 * @param page The page number of results
		 * @param sort Sorting order for results, can be one of: name-desc, name-asc, distance-asc, distance-desc
		 * 
		 */
		public function findVenues (search_text:String = null, location:String = null, radius:Number = NaN, country:Country = null, state:State = null, metro:Metro = null, per_page:Number = NaN, page:Number = NaN, sort:String = null):void {
			var params:String = "";

			
			if (search_text != null && search_text != '') {
				params += "&search_text=" + escape(search_text);			
			}
			
			if (location != null && location != '') {
				params += "&location=" + escape(location);
			}
			
			if (!isNaN(radius)) {
				params += "&radius=" + radius;
			}
			
			if (country != null) {
				params += "&country_id=" + country.id;
			} 
			
			if (state != null) {
				params += "&state_id=" + state.id;
			}
			
			if (metro != null) {
				params += "&metro_id=" + metro.id;
			}
			
			if (!isNaN(per_page)) {
				params += "&per_page=" + per_page;
			}
			
			if (!isNaN(page)) {
				params += "&page=" + page;
			}
		
		    if (sort != null) {
		    	params += "&sort=" + sort;
		    }
			sendQuery("venue.search", (params), UpcomingResultEvent.VENUE_SEARCH, 'GET');			
		}  

		/**
		 * This method requests a venue list for a particular metro area.
		 * When the list is returned, the current instance of UpcomingService dispatches a
		 * <code>UpcomingResultEvent.VENUE_GET_LIST</code> event.
		 * 
		 * @param metro The metro area to search for venues.
		 * 
		 */		
		public function getVenueList (metro:Metro) : void {
			var params:String = "";
			if (token != null) {
			params += "token=" + escape(token);
			}
			params += "&metro_id=" + metro.id;
			
			sendQuery("venue.getList", (params), UpcomingResultEvent.VENUE_GET_LIST, 'GET');		

		}
		
		/**
		 * This method requests a list of all available event categories.
		 * The categories are pre-defined by Upcoming.org. When the list of categories
		 * is loaded, the current instance of UpcomingService dispatches a
		 * <code>UpcomingResultEvent.CATEGORY_GET_LIST</code> event.
		 * 
		 */		
		public function getCategoryList () : void {
			sendQuery("category.getList", null, UpcomingResultEvent.CATEGORY_GET_LIST, 'GET');
		}
		
		/**
		 * This method retrieves a list of events currently watched by the authenticated user.
		 * The events are accompanied by watchlist ids and user's status ("watching" or "attending").
		 * When the list is returned, the current instance of UpcomingService issues an
		 * <code>UpcomingResultEvent.WATCHLIST_GET_LIST</code> event.
		 * @param min_date Minimum date for the list range.
		 * @param max_date Maximum date for the list range.
		 * @param sort Sorting order for the results. Can be one of: 'start-date-asc', 'start-date-desc', 'post-date-asc', 'post-date-desc'
		 * 
		 */		
		public function getMyWatchList (min_date:Date = null, max_date:Date = null, sort:String = null) : void {
			
			var params:String = "";
			params = params + "token=" + token;
			if (min_date != null) {
			params = params + "&min_date=" + stringFromDate(min_date);
			}
			if (max_date != null) {
			params = params + "&max_date=" + stringFromDate(max_date);
			}
			if (sort != null && sort != '') {
			params = params + "&sort=" + sort;
			}
			
			sendQuery("watchlist.getList", params, UpcomingResultEvent.WATCHLIST_GET_LIST, 'GET');		
			
		}
		
		/**
		 * This method adds a specified event to the user's watchlist, 
		 * and assigns the user's status for this event (either "watch" or "attend").
		 * The event must already exist in the Upcoming.org database.
		 * When the adding is complete, the current instance of UpcomingService dispatches
		 * a <code>UpcomingResultEvent.WATCHLIST_ADDED</code> event.
		 * @param uevent The event to add to the user's watchlist.
		 * @param status The status of the user's attendance (Either "watch", or "attend").
		 */		
		public function addToWatchList (uevent:uEvent, status:String = null) : void {
			var params:String = "";
			params = params + "token=" + token;
			params = params + "&event_id" + uevent.id;
			
			if (status != null && status != '') {
			params = params + "&status=" + status;
			}
			
			sendQuery("watchlist.add", params, UpcomingResultEvent.WATCHLIST_ADDED, 'POST');
		}
		
		/**
		 * This method removes the user from a watchlist, given a watchlist id. Once
		 * the removal is complete, the current instance of UpcomingService dispatches
		 * a <code>UpcomingResultEvent.WATCHLIST_REMOVED</code> event.
		 * @param watchlist_id The id of the watchlist to remove the user from.
		 * 
		 */	
		public function removeFromWatchList (watchlist_id:Number) : void {
			var params:String = "";
			params = params + "token=" + token;
			params = params + "&watchlist_id=" + watchlist_id;
			
			sendQuery("watchlist.remove", params, UpcomingResultEvent.WATCHLIST_REMOVED, 'POST');
		}
		
		
		/**
		 * This method returns all metro areas for a specific state.
		 * The state is required to have the id field filled in.
		 * When the list is returns, the current instance of UpcomingService
		 * dispatches a <code>UpcomingResultEvent.METRO_GET_LIST</code> event.
		 * @param state The state to get a list of metro areas for.
		 * 
		 */		
		public function getMetroListByState(state:State):void {
			var params:String = "state_id=" + state.id;
			sendQuery("metro.getList", params, UpcomingResultEvent.METRO_GET_LIST, 'GET', state); 	
		}
		
		/**
		 * This method returns a list of all metro areas that the currently
		 * authenticated user is subscribed to. A valid token is necessary
		 * to get the results. Once the results are returns, the current instance
		 * of UpcomingService dispatches a <code>UpcomingResultEvent.METRO_GET_MY_LIST</code> event.
		 * 
		 */		
		public function getMyMetroList(): void {
			var params:String = "token=" + token;
			sendQuery("metro.getMyList", params, UpcomingResultEvent.METRO_GET_MY_LIST, 'GET'); 	
		}
		
		/**
		 * This method returns one metro area closest to the specified latitude
		 * and longitude. When the results are returns, the current instance
		 * of UpcomingService dispatches an <code>UpcomingResultEvent.METRO_GET_LAT_LON</code> event.
		 * @param latitude The latitude in proximity to a metro.
		 * @param longitude The longitude in proximity to a metro.
		 * 
		 */		
		public function getMetroByLatLon(latitude:Number, longitude:Number): void {
			var params:String = "latitude=" + latitude;
			params = params + "&longitude=" + longitude;
			
			sendQuery("metro.getForLatLon", params, UpcomingResultEvent.METRO_GET_LAT_LON, 'GET');
		}
		
		/**
		 * This method returns the list of all countries in the Upcoming.org database.
		 * When the results are ready, the current instance of UpcomingService dispatches
		 * <code>UpcomingResultEvent.GET_COUNTRY_LIST</code> event.
		 * 
		 */		
		public function getCountryList():void {
			sendQuery("metro.getCountryList", null, UpcomingResultEvent.GET_COUNTRY_LIST, 'GET');
		}
		
		/**
		 * This method returns the list of all states in a given country.
		 * When the results are ready, the current instance of UpcomingService dispatches
		 * <code>UpcomingResultEvent.GET_STATE_LIST</code> event.
		 * @param country The country for which the list of states is retrieved.
		 * 
		 */		
		public function getStateList (country:Country) : void {
			sendQuery("metro.getStateList", null, UpcomingResultEvent.GET_STATE_LIST, 'GET');
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
		private function sendQuery(method:String, params:String, dispatchType:String, methodType:String = 'POST', additionalData:* = null) : void {
			
		var sendQuery:String;
			sendQuery = (api_URL);
		var queryParams:String;
		
			if(params != null) {
				queryParams = ("api_key=" + api_id + "&method=" + method + "&"+params);
			} else {
				queryParams = ("api_key=" + api_id + "&method=" + method);
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
				var result:Array = new Array();
				
				queryXML = new XML(queryLoader.data);
				switch(dispatchType) {
					case UpcomingResultEvent.EVENT_SEARCH:
						for each (resultNode in queryXML.children()) {
						var new_event_state:State = new State(_self, resultNode.@venue_state_id, resultNode.@venue_state_name, resultNode.@venue_state_code);
						var new_metro:Metro = new Metro(_self, resultNode.@metro_id, null, null, new_event_state);
						var new_event_venue:Venue = new Venue(_self, resultNode.@venue_name, resultNode.@venue_address, resultNode.@venue_city, new_metro, resultNode.@venue_id, resultNode.@venue_zip, resultNode.@venue_phone, resultNode.@venue_url, null, false); 				
						var uE:uEvent = new uEvent(_self);
						uE.id = resultNode.@id;
						uE.name = resultNode.@name;
						uE.description = resultNode.@description;
						uE.start_date = dateFromString(resultNode.@start_date, resultNode.@start_time);
						if (resultNode.@end_date != "0000-00-00") {
								uE.end_date = dateFromString(resultNode.@end_date, resultNode.@end_date);
						}
						uE.post_date = dateFromString(resultNode.@date_posted.split(" ")[0], resultNode.@date_posted.split(" ")[1]);
						uE.venue = new_event_venue;
						uE.personal = resultNode.@personal==1?true:false;
						uE.selfpromotion = resultNode.@selfpromotion==1?true:false;
						uE.user = new User(_self, resultNode.@user_id, null, null, null, null);
						uE.category = new Category(_self, resultNode.@category_id, null, null);
						uE.latitude = resultNode.@latitude;
						uE.longitude = resultNode.@longitude;
						uE.geocoding_precision = resultNode.@geocoding_precision;
						uE.geocoding_ambiguous = resultNode.@geocoding_ambiguous==1?true:false;
						result.push(uE);
						}
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.EVENT_SEARCH, result)); 
						break;
					case UpcomingResultEvent.METRO_GET_LIST:
						var mystate:State = additionalData;
						for each (resultNode in queryXML.children()) {					
						var mo_list:Metro = new Metro(_self, resultNode.@id, resultNode.@name, resultNode.@code, mystate);
						result.push(mo_list);
						}			
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.METRO_GET_LIST, result));
						break;
						
					case UpcomingResultEvent.METRO_GET_MY_LIST:
						for each (resultNode in queryXML.children()) {					
						var mo_mylist:Metro = new Metro(_self, resultNode.@id, resultNode.@name, resultNode.@code, new State(_self, resultNode.@state_id, resultNode.@state_name, resultNode.@state_code));
						result.push(mo_mylist);
						}			
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.METRO_GET_MY_LIST, result));
						break;
						
					case UpcomingResultEvent.METRO_GET_LAT_LON:
						for each (resultNode in queryXML.children()) {					
						var mo_latlon:Metro = new Metro(_self, resultNode.@id, resultNode.@name, resultNode.@code, new State(_self, resultNode.@state_id, resultNode.@state_name, resultNode.@state_code));
						result.push(mo_latlon);
						}			
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.METRO_GET_LAT_LON, result));
						break;
						
 					case UpcomingResultEvent.METRO_SEARCH:
						for each (resultNode in queryXML.children()) {					
						var mo_search:Metro = new Metro(_self, resultNode.@id, resultNode.@name, resultNode.@code, new State(_self, Number(resultNode.@state_id), null, null));
						result.push(mo_search);
						}			
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.METRO_SEARCH, result));
						break;
 						
 					case UpcomingResultEvent.VENUE_SEARCH:
						for each (resultNode in queryXML.children()) {
							var venue:Venue = new Venue(_self);
							venue.id = resultNode.@id;
							venue.name = resultNode.@name;
							venue.address = resultNode.@address;
							venue.city = resultNode.@city;
							venue.zip = resultNode.@zip;
							venue.url = resultNode.@url;
							venue.description = resultNode.@description;
							venue.user = new User(_self, resultNode.@user_id, null, null, null, null);
							venue.privacy = resultNode.@private=="0"?false:true;
							venue.latitude = resultNode.@latitude;
							venue.longitude = resultNode.@longitude;
							venue.geocoding_precision = resultNode.@geocoding_precision;
							venue.geocoding_ambiguous = resultNode.@geocoding_ambiguous;
							result.push(venue);
						}		
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.VENUE_SEARCH, result));
						break;
						
					case UpcomingResultEvent.VENUE_GET_LIST:
						for each (resultNode in queryXML.children()) {
							var new_venue:Venue = new Venue(_self);
							new_venue.id = resultNode.@id;
							new_venue.name = resultNode.@name;
							new_venue.address = resultNode.@address;
							new_venue.city = resultNode.@city;
							new_venue.zip = resultNode.@zip;
							new_venue.url = resultNode.@url;
							new_venue.description = resultNode.@description;
							new_venue.user = new User(_self, resultNode.@user_id, null, null, null, null);
							new_venue.privacy = resultNode.@private=="0"?false:true;
							new_venue.latitude = resultNode.@latitude;
							new_venue.longitude = resultNode.@longitude;
							new_venue.geocoding_precision = resultNode.@geocoding_precision;
							new_venue.geocoding_ambiguous = resultNode.@geocoding_ambiguous=="0"?false:true;
							result.push(new_venue);
						}		
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.VENUE_GET_LIST, result));
						break;
						
					case UpcomingResultEvent.CATEGORY_GET_LIST:
						for each (resultNode in queryXML.children()) {					
						var new_cat:Category = new Category(_self, resultNode.@id, resultNode.@name, resultNode.@description);
						result.push(new_cat);
						}			
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.CATEGORY_GET_LIST, result));
						break;
						
					case UpcomingResultEvent.WATCHLIST_GET_LIST:
						for each (resultNode in queryXML.children()) {					
						var new_event:uEvent = new uEvent(_self);
						new_event.id = resultNode.@event_id;
  						result.push({event: new_event, watchlist_id: resultNode.@id, status: resultNode.@status});
						}			
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.WATCHLIST_GET_LIST, result));
						break;
					
					case UpcomingResultEvent.WATCHLIST_ADDED:
						for each (resultNode in queryXML.children()) {					
  						result.push({watchlist_id: resultNode.@id});
						}			
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.WATCHLIST_ADDED, result));
						break;
						
					case UpcomingResultEvent.WATCHLIST_REMOVED:
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.WATCHLIST_REMOVED, null));
						break;
						
					case UpcomingResultEvent.GET_COUNTRY_LIST:
						for each (resultNode in queryXML.children()) {					
						var new_country:Country = new Country(_self, resultNode.@id, resultNode.@name, resultNode.@code);
						result.push(new_country);
						}			
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.GET_COUNTRY_LIST, result));
						break;
						
					case UpcomingResultEvent.GET_STATE_LIST:
						for each (resultNode in queryXML.children()) {					
						var new_state:State = new State(_self, resultNode.@id, resultNode.@name, resultNode.@code);
						result.push(new_state);
						}			
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.GET_STATE_LIST, result));
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