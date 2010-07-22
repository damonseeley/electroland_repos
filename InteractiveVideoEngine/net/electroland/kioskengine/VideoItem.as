package net.electroland.kioskengine {
	import flash.events.*;
	import flash.net.NetConnection;
	import flash.net.NetStream;
	import flash.media.Video;
	import flash.display.Stage;
	import flash.media.SoundTransform;
	
	public class VideoItem extends Item {
		private var nc:NetConnection;
		private var ns:NetStream;
		private var netClient:Object = new Object();
		private var vid:Video;
		private var skipToPoint:int;
		private var filename:String;
		private var loaded:Boolean = false;
		private var playedCompletely:Boolean = false;
		
		/*
		
		VIDEOITEM.as
		by Aaron Siegel, 7-4-09
		
		*/
		
		public function VideoItem(conductor:Conductor, itemElement:XML){
			super(conductor);
			parseXML(itemElement);
			//alpha = 1;
		}
		
		public function loadContent():void{
			nc = new NetConnection();
			nc.connect(null);
			ns = new NetStream(nc);
			vid = new Video(getWidth(), getHeight());	// should scale to stage
			addChild(vid);
			vid.attachNetStream(ns); 
			ns.addEventListener(NetStatusEvent.NET_STATUS, netstat);
			
			if(getHorizontalAlign() == "center"){
				vid.x = 0 - vid.width/2;
			} else if(getHorizontalAlign() == "right"){
				vid.x = 0 - vid.width;
			}
			if(getVerticalAlign() == "center"){
				vid.y = 0 - vid.height/2;
			} else if(getVerticalAlign() == "right"){
				vid.y = 0 - vid.height;
			}
			
			if(itemEntries.hasOwnProperty("all")){
				filename = itemEntries["all"].value;
			} else {
				filename = itemEntries[conductor.getLanguage()].value;
			}
			ns.play(filename);
			var videoVolumeTransform:SoundTransform = new SoundTransform();
			videoVolumeTransform.volume = conductor.getGain();
			ns.soundTransform = videoVolumeTransform;
			netClient.onMetaData = function(meta:Object){ netStreamHandler(meta); };
			ns.client = netClient;
			//trace("default buffer time: "+ns.bufferTime);
			//ns.bufferTime = 2;
			// old contentLoadedCallback location
		}
		
		public function mute():void{
			if(ns != null){
				var videoVolumeTransform:SoundTransform = new SoundTransform();
				videoVolumeTransform.volume = 0;
				ns.soundTransform = videoVolumeTransform;
			}
		}
		
		public function netstat(stats:NetStatusEvent):void{
			//trace(stats.info.code);
			/*
			if(ns.bufferLength > 0){
				if(!loaded){
					// video has loaded and started, so NOW call back
					if(contentLoadedCallback != null){
						contentLoadedCallback(this);	// tell unit this has been loaded
					}
					loaded = true;
				}
			}
			*/
			if(stats.info.code == "NetStream.Buffer.Full"){
				
				if(!loaded){
					// video has loaded and started, so NOW call back
					if(contentLoadedCallback != null){
						contentLoadedCallback(this);	// tell unit this has been loaded
					}
					conductor.logItemEvent(" ", " ", getName(), getDescription(), "video started");
					loaded = true;
				}
				
			} else if(stats.info.code == "NetStream.Play.Stop"){
				// dispatch event to unit that video file has ended
				if(itemEntries.hasOwnProperty("all")){
					if(itemEntries["all"].looping){
						ns.seek(0);
					} else {
						dispatchEvent(new ItemEvent("videoComplete", "default"));
						playedCompletely = true;
					}
				} else {
					if(itemEntries[conductor.getLanguage()].looping){
						ns.seek(0);						
					} else {
						dispatchEvent(new ItemEvent("videoComplete", "default"));
						playedCompletely = true;
					}
				}
			} else if(stats.info.code == "NetStream.Play.StreamNotFound"){
				trace("ERROR: video file "+filename+" not found");
			}
		}
		
		public function netStreamHandler(meta:Object):void{
			/*
			for (var propName:String in meta) {
				trace(propName + " = " + meta[propName]);
			}
			*/
			skipToPoint = meta.duration - 5;		// 5 seconds from the end
			conductor.parent.stage.addEventListener(KeyboardEvent.KEY_DOWN, onKeyPressedEvent);
			//trace(skipToPoint);
		}
		
		public function onKeyPressedEvent(event:KeyboardEvent):void{
			//trace(event.keyCode);
			if(event.keyCode == 39){	// right arrow key
				//trace("skip forward");
				ns.seek(skipToPoint);
			}
		}
		
		public function pause():void{
			if(ns != null){
				ns.pause();
			}
		}
		
		public function play():void{
			if(ns != null){
				ns.resume();
			}
		}
		
		override public function reset():void{
			trace("VIDEO stop requested");
			if(vid != null){
				removeChild(vid);
				ns.close();
				vid.clear();
				loaded = false;
				trace("VIDEO STOPPED");
				if(playedCompletely){
						conductor.logItemEvent(" ", " ", getName(), getDescription(), "video complete");					
				} else {
					conductor.logItemEvent(" ", " ", getName(), getDescription(), "video interrupted");
				}
				conductor.parent.stage.removeEventListener(KeyboardEvent.KEY_DOWN, onKeyPressedEvent);
			}
			super.reset();
		}
		
		public function startTimer():void{
			// begin timers on various effects
			startEffectTimers();
		}
		
	}
	
}