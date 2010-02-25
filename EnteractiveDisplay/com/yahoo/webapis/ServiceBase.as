/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis
{
	import flash.events.EventDispatcher;
	import flash.events.IEventDispatcher;

	/**
	 * The base class for Service methods
	 * 
	 * @langversion ActionScript 3.0
	 * @playerversion Flash 9
	 * @author Alaric Cole 02/22/07
	 */
	
	public class ServiceBase extends EventDispatcher
	{
		/**
		 * An Application ID is a string that uniquely identifies your application. Think of it as like a User-Agent string. 
		 * If you have multiple applications, you must use a different ID for each one. 
		 * @see http://developer.yahoo.com/faq/index.html#appid
		 */
		 //TODO: dispatch error if no applicationId is set
		public var applicationId:String;
		
		
		public function ServiceBase(target:IEventDispatcher=null)
		{
			
			super(target);
		}
		
	}
}

