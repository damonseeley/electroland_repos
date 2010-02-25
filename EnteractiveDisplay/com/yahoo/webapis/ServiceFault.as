/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis
{
	/**
	* Error Event class in response to fault events from the Yahoo! APIs.
	* 
	* @langversion ActionScript 3.0
	* @playerversion Flash 9
	* @author Alaric Cole 02/22/07
	*/
	public class ServiceFault extends Error
	{
		private var _faultCode:String;
		private var _faultString:String;
		private var _faultDetail:String;
		
		/**
		 * The fault code.
		 */
		public function get faultCode():String
		{
			return _faultCode;
		}
		
		public function set faultCode( value:String ):void
		{
			_faultCode = value;
		}
		
		/**
		 * The fault string.
		 */
		public function get faultString():String
		{
			return _faultString;
		}
		
		public function set faultString( value:String ):void
		{
			_faultString = value;
		}
		
		/**
		 * The fault detail.
		 */
		public function get faultDetail():String
		{
			return _faultDetail;
		}
		
		public function set faultDetail( value:String ):void
		{
			_faultDetail = value;
		}
		
		
		public function ServiceFault(faultCode:String, faultString:String, faultDetail:String = null)
		{
			this.faultCode = faultCode;
			this.faultString = faultString;
			this.faultDetail = faultDetail;
			super(faultCode);
		}
		
		
		
	}
}