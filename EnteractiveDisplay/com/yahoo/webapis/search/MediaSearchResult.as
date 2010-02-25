/*
Copyright (c) 2008 Yahoo! Inc.  All rights reserved.  
The copyrights embodied in the content of this file are licensed under the BSD (revised) open source license
*/
package com.yahoo.webapis.search
{

	/**
	 * MediaSearchResult is a Value Object for the Search API.
	 * @langversion ActionScript 3.0
	 * @playerversion Flash 9
	 * @author Alaric Cole 02/22/07
	 * 
	 */
	public class MediaSearchResult extends SearchResult
	{
		
		//private var _title:String;
		private var _summary:String;
	
		//private var _category:String;
		
		private var _url:String;
		private var _clickURL:String;
		private var _referrerURL:String;
		
		private var _thumbnail:Thumbnail;
		
		//private var _modificationDate:Date;
	
		private var _fileSize:uint;
		private var _fileFormat:String;
		
		private var _height:Number;
		private var _width:Number;
		
		private var _publisher:String;
		private var _restrictions:String;
		private var _copyright:String;
		
		private var _duration:Number;
		private var _isStreaming:Boolean;
		private var _numChannels:uint;
		
		private var _price:Number;
		private var _quality:Number;
		
		/**
		 * Construct a new MediaSearchResult instance.
		 *
		 */
		public function MediaSearchResult()
		{
			// Do Nothing.
		}
		
		
		
		/**
		 * The title.
		 */
		
		
		
		
		/**
		 * The summary.
		 */
		public function get summary():String
		{
			return _summary;
		}
		
		public function set summary( value:String ):void
		{
			_summary = value;
		}
		
		/**
		 * The thumbnail.
		 */
		public function get thumbnail():Thumbnail
		{
			return _thumbnail;
		}
		
		public function set thumbnail( value:Thumbnail ):void
		{
			_thumbnail = value;
		}
		
		
		
			
		/**
		 * The  url.
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
		 * The click URL.
		 */
		public function get clickURL():String
		{
			return _clickURL;
		}
		
		public function set clickURL( value:String ):void
		{
			_clickURL = value;
		}
		
		/**
		 * The display URL.
		 */
		public function get referrerURL():String
		{
			return _referrerURL;
		}
		
		public function set referrerURL( value:String ):void
		{
			_referrerURL = value;
		}
		
			
		
		/**
		 * The file size.
		 */
		 //The following two properties could be put into the flash.net.FileReference class or a custom File class
		public function get fileSize():uint
		{
			return _fileSize;
		}
		
		public function set fileSize( value:uint ):void
		{
			_fileSize = value;
		}
		
		/**
		 * The file format: one of bmp, gif, jpg, or png.
		 */
		public function get fileFormat():String
		{
			return _fileFormat;
		}
		
		public function set fileFormat( value:String ):void
		{
			_fileFormat = value;
		}
		
		
		/**
		 * The height in pixels.
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
		 * The width in pixels.
		 */
		public function get width():Number
		{
			return _width;
		}
		
		public function set width( value:Number ):void
		{
			_width = value;
		}
		
		/**
		 * The publisher of the content.
		 */
		public function get publisher():String
		{
			return _publisher;
		}
		
		public function set publisher( value:String ):void
		{
			_publisher = value;
		}
		
		/**
		 * The restrictions.
		 */
		public function get restrictions():String
		{
			return _restrictions;
		}
		
		public function set restrictions( value:String ):void
		{
			_restrictions = value;
		}
		
		/**
		 * The copyright of the ImageSearchResult.
		 */
		public function get copyright():String
		{
			return _copyright;
		}
		
		public function set copyright( value:String ):void
		{
			_copyright = value;
		}
	
		/**
		 * The duration in seconds.
		 */
		public function get duration():Number
		{
			return _duration;
		}
		
		public function set duration( value:Number ):void
		{
			_duration = value;
		}
		
		/**
		 * Whether the media is streaming or not; one of bmp, gif, jpg, or png.
		 */
		public function get isStreaming():Boolean
		{
			return _isStreaming;
		}
		
		public function set isStreaming( value:Boolean ):void
		{
			_isStreaming = value;
		}
		
		/**
		 * The number of channels.
		 */
		public function get numChannels():uint
		{
			return _numChannels;
		}
		
		public function set numChannels( value:uint ):void
		{
			_numChannels = value;
		}
		
		/**
		 * The price provider.
		 */
		public function get price():Number
		{
			return _price;
		}
		
		public function set price( value:Number ):void
		{
			_price = value;
		}
		
		/**
		 * The quality of the media from 1 to 5.
		 */
		public function get quality():Number
		{
			return _quality;
		}
		
		public function set quality( value:Number ):void
		{
			_quality = value;
		}
	}
}