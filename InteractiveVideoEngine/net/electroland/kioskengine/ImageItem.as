package net.electroland.kioskengine {
	import flash.net.URLRequest;
	import flash.display.Loader;
	import flash.events.*;
		
	public class ImageItem extends Item {
		private var loader:Loader = new Loader();
		
		/*
	
		IMAGEITEM.as
		by Aaron Siegel, 7-3-09
		
		Simple wrapper around flash image drawing.
		
		*/
		
		public function ImageItem(conductor:Conductor, itemElement:XML){
			super(conductor);
			parseXML(itemElement);
			loader.contentLoaderInfo.addEventListener(Event.COMPLETE, imageComplete);
			loader.contentLoaderInfo.addEventListener(IOErrorEvent.IO_ERROR, ioErrorHandler);
		}
		
		private function imageComplete(event:Event):void {
			addChild(loader);
			loader.width = getWidth();
			loader.height = getHeight();
			if(getHorizontalAlign() == "center"){
				loader.x = 0 - loader.width/2;
			} else if(getHorizontalAlign() == "right"){
				loader.x = 0 - loader.width;
			}
			if(getVerticalAlign() == "center"){
				loader.y = 0 - loader.height/2;
			} else if(getVerticalAlign() == "bottom"){
				loader.y = 0 - loader.height;
			}
			if(contentLoadedCallback != null){
				contentLoadedCallback(this);	// tell unit this has been loaded
			}
		}
		
		private function ioErrorHandler(event:Event):void {
			if(itemEntries.hasOwnProperty("all")){
				trace("ERROR: failed to load url "+itemEntries["all"].value);
			} else {
				trace("ERROR: failed to load url "+itemEntries[conductor.getLanguage()].value);
			}
		}
		
		public function loadContent():void{
			if(itemEntries.hasOwnProperty("all")){
				loader.load(new URLRequest(itemEntries["all"].value));	// initial image load
			} else {
				loader.load(new URLRequest(itemEntries[conductor.getLanguage()].value));	// initial image load
			}
		}
		
		public function startTimer():void{
			// begin timers on various effects
			startEffectTimers();
		}
		
	}
}