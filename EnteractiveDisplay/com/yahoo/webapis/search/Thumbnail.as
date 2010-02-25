/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.search
{
	[Bindable]
	/**
	 * Thumbnail is a Value Object for the Search API.
	 * @langversion ActionScript 3.0
	 * @playerversion Flash 9
	 * @author Alaric Cole 02/22/07
	*/
	public class Thumbnail extends Object
	{
		
		
		private var _url:String;
		
		private var _height:Number;
		private var _width:Number;
		
		
		/**
		 * Construct a new Thumbnail instance.
		 *
		 */
		public function Thumbnail(url:String = null)
		{
			this.url = url;
		}
		
		
		
		/**
		 * The url to the Thumbnail.
		 */
		public function get url():String
		{
			return _url;
		}
		
		public function set url( value:String ):void
		{
			_url = value;
		}
		
		
		
		/**
		 * The height of the Thumbnail in pixels.
		 */
		public function get height():Number
		{
			return _height;
		}
		
		public function set height( value:Number ):void
		{
			_height = value;
		}
		
		/**
		 * The width of the Thumbnail in pixels.
		 */
		public function get width():Number
		{
			return _width;
		}
		
		public function set width( value:Number ):void
		{
			_width = value;
		}
		
	}
}