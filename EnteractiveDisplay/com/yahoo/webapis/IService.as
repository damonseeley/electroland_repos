/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis
{

	/**
	 * The interface provides the methods and properties that a typical Yahoo! service requires.
	 * All services should implement this interface.
	 * 
	 * @langversion ActionScript 3.0
	 * @playerversion Flash 9
	 * @author Alaric Cole 02/22/07
	 */
	
	public interface IService
	{
		/**
		 * This provides a way to use a standard method for invoking a service.
		 */
		function send(parameters:Object = null):void;
		
		/**
		* The result of the last service invocation
		*/
		function get lastResult():Object;
		

	}
}

