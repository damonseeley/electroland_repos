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
	import com.yahoo.webapis.upcoming.events.*;
	
	/**
	* Yahoo! Upcoming.org API Category Class. This class is used to hold information
	* about event categories used in Upcoming.org classification.
	* 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Scott Morgan 02/25/07
	*/
	public class Category extends EventDispatcher
	{
		/**
		* Defines the parent UpcomingService for this category (used to access the token and API key)
		*/
 		public var parent_service:UpcomingService;
 		/**
 		* The Upcoming.org category ID.
 		*/ 		
 		public var id:Number;
 		/**
 		* Upcoming.org category name.
 		*/ 		
 		public var name:String;
 		/**
 		* Upcoming.org category description
 		*/ 		
 		public var description:String;

 		/**
 		 * Category class constructor.
 		 * @param _parent_service An UpcomingService with an instantiated token and API key.
 		 * @param _id Upcoming.org category identification number
 		 * @param _name Upcoming.org category name
 		 * @param _description Upcoming.org category description
 		 */
 		public function Category (_parent_service:UpcomingService, _id:Number = NaN, _name:String = null, _description:String = null) {
 			parent_service = _parent_service;
 			id = _id;
 			name = _name;
 			description = _description;
 		}
		
	
	}
} 