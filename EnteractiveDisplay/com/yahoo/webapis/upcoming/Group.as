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
	* Yahoo! Upcoming.org API Group Class. This class is used to hold
	* information about Upcoming.org user groups.
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Scott Morgan and Allen Rabinovich 02/25/07
	*/
	public class Group extends EventDispatcher
	{
		/**
		* Defines the parent UpcomingService for this country (used to access the token and API key)
		*/
		public var parent_service:UpcomingService;		
		/**
		* The Upcoming.org user group identification number.
		*/		
		public var id:Number;
 		/**
 		* Upcoming.org user group name.
 		*/		
 		public var name:String;
 		/**
 		* Upcoming.org user group description.
 		*/ 		
 		public var description:String;
 		/**
 		* Upcoming.org group moderation level. Can be either 'moderated' or 'unmoderated'.
 		*/ 		
 		public var moderation_level:String;
 		/**
 		* Upcoming.org group privacy setting. Can be either <code>true</code>(private), or <code>false</code>(public).
 		*/ 		
 		public var is_private:Boolean;
 		/**
 		* The group's creator.
 		*/ 		
 		public var owner_user:User;
 		/**
 		* Group's commercial account ID, if present
 		*/ 		
 		public var commercial_account_id:Number;
		/**
		* Number of members in the group.
		*/ 		
		public var num_members:Number;
		/**
		* The date and time when the group has been started.
		*/		
		public var start_date:Date;
	    /**
	    * Group's privacy setting. If <code>true</code>, then the group is private; if <code>false</code>, then public.
	    */
	    public var personal:Boolean;
 	    /**
 	    * The Metro area associated with the group.
 	    */
 	    public var metro_id:Metro;

 		/**
 		 * Group class constructor
 		 * @param _parent_service An UpcomingService with an instantiated token and API key.
 		 * @param _name The name of the user group.
 		 * @param _description Group description
 		 * @param _moderation_level Group moderation level, can be either "moderated" or "unmoderated"
 		 * @param _is_private Group privacy setting (<code>true</code> if private, <code>false</code>otherwise)
 		 * @param _owner_user The User that owns the group.
 		 * @param _commercial_account_id The group's commercial account ID, if any.
 		 * @param _num_members The number of members in the group.
 		 * @param _id The group id.
 		 * @return 
 		 * 
 		 */		
 		public function Group (_parent_service:UpcomingService, _name:String = null, _description:String = null, _moderation_level:String = null, _is_private:Number = 1, _timestamp:String = null, _owner_user:User = null, _commercial_account_id:Number = undefined, _num_members:Number = undefined, _id:Number = undefined) {
	 		parent_service = _parent_service;
	 		id = _id;
	 		name = _name;
	 		description = _description;
	 		moderation_level = _moderation_level;
	 		is_private = _is_private == 1 ? true:false;
	 		owner_user = _owner_user;
	 		commercial_account_id = _commercial_account_id;
			num_members = _num_members;
 		}
		
 		/**
 		 * This method updates the data in the specific instance of Group
 		 * by the ID number. If the <code>id</code> property is already filled in,
 		 * then no argument needs to be passed. Otherwise, pass the group's
 		 * Upcoming.org identification number.
 		 * <p>When the update completes, the Group class will dispatch an
 		 * <code>UpcomingResultEvent.</code> event.</p>
 		 * @param _id The country's Upcoming.org identification number, if
 		 * none is filled in.
 		 * 
 		 */ 	
		public function update(_id:Number = NaN):void {
			if (!isNaN(_id)) {
				id = _id;
			}
			if (parent_service.token != null) {
				sendQuery("group.getInfo", ("group_id=" + id + "&token=" + parent_service.token), UpcomingResultEvent.GROUP_LOADED, 'GET');
			} else {
				sendQuery("group.getInfo", ("group_id=" + id), UpcomingResultEvent.GROUP_LOADED, 'GET');
			}			
		}
		
 		/**
 		 * This method adds the Group instance for which it's called
 		 * to the Upcoming.org database.
 		 * <p>Upcoming.org requires the group to have at least a name.
 		 * If the name is already taken, the server returns a 409 error.
 		 * 
 		 */ 	
		public function add():void {
			var params:String = "";
			params += "token=" + escape(parent_service.token);
			params += "&name=" + escape(name);
			if (description != null && description != '') {
				params += "&description=" + escape(description);
			}
			if (moderation_level != null && moderation_level != '') {
				params += "&moderation_level=" + escape(moderation_level);
			} else {
				params += "&moderation_level=" + escape(moderation_level);
			}
			var value:Number = is_private == true ? 1:0;
			params += "&is_private=" + value.toString();

			sendQuery("group.add", (params), UpcomingResultEvent.GROUP_ADDED, "POST");
		}
		
		/**
		 * This method commits edits for an existing Group to the
		 * Upcoming.org database. It requires that the current
		 * group have a valid group ID and a group name.
		 */
		public function commit():void {
			
			if (isNaN(id)) {
				add();
			} else {
				var params:String = "";
				params += "token=" + escape(parent_service.token);
				params += "&group_id=" + id;
				params += "&name=" + escape(name);
				if (description != null && description != '') {
					params += "&description=" + escape(description);
				}
				if (moderation_level != null && moderation_level != '') {
					params += "&moderation_level=" + escape(moderation_level);
				} else {
					params += "&moderation_level=" + escape(moderation_level);
				}
				var value:Number = is_private == true ? 1:0;
				params += "&is_private=" + value;
	
				sendQuery("group.edit", (params), UpcomingResultEvent.GROUP_EDITED, "POST");
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

			var sendQuery:String = (parent_service.api_URL);
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
				var resultObject:Object = new Object();
				
				queryXML = new XML(queryLoader.data);
				switch(dispatchType) {
					case UpcomingResultEvent.GROUP_LOADED:
						for each (resultNode in queryXML.children()) {
							id = resultNode.@id;
	 						name = resultNode.@name;
	 						description = resultNode.@description;
	 						moderation_level = resultNode.@moderation_level;
	 						is_private = resultNode.@is_private==0?false:true;
	 						owner_user = new User(parent_service, resultNode.@owner_user_id, null, null, null, null);
	 						commercial_account_id = resultNode.@commercial_account_id;
							num_members = resultNode.@num_members;
						}
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.GROUP_LOADED, this)); 
						break;
					case UpcomingResultEvent.GROUP_ADDED:
						for each (resultNode in queryXML.children()) {
							id = resultNode.@id;
	 						name = resultNode.@name;
	 						description = resultNode.@description;
	 						start_date = formatDate(resultNode.@start_date, resultNode.@start_time);
						    personal = resultNode.@personal;
						    metro_id = resultNode.@metro_id;
						}
						dispatchEvent(new UpcomingResultEvent(UpcomingResultEvent.GROUP_ADDED, this)); 
						break;
					
 					}
				}
				
			
				function xmlError(evtObj:Object):void {
					
		        	var error:UpcomingErrorEvent = new UpcomingErrorEvent(UpcomingErrorEvent.XML_LOADING, evtObj);
					dispatchEvent(error);
		        }
			}
			
			private function formatDate(myDate:*, myTime:* = '00:00:00'):Date {
				if (myDate is String) {
					var edate:Array = myDate.split("-");
	                var etime:Array = myTime.split(":");
	                var newDate:Date = new Date();
	                newDate.setFullYear(Number(edate[0]), Number(edate[1])-1, Number(edate[2]));
	                newDate.setHours(Number(etime[0]), Number(edate[1]), Number(edate[2]));
	   			}
                
                return newDate;
			}

		}
		
	}