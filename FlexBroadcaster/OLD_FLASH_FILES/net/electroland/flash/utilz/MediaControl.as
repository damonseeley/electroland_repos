﻿package net.electroland.flash.utilz {		import flash.display.*;	import flash.media.Video;	import flash.net.URLRequest;	import flash.net.NetConnection;	import flash.net.NetStream;	import flash.events.*;	import flash.events.TimerEvent;	import flash.utils.Timer;	public class MediaControl extends MovieClip{		private var vidSWF:Loader;		private var contentSWF:Loader;		private var mcVid:MovieClip;		private var nc:NetConnection;		private var ns:NetStream;		private var duration:Number;		private var client:CustomClient;		private var vidSrc:String;		private var playbackTimer:Timer;		private var stVidHolderSwf:String = "fpoVidHolder.swf";		private var rootmc:DisplayObjectContainer;		private var xposition:Number;		private var yposition:Number;		private var type:String;				public function MediaControl(xpos:Number,ypos:Number) {			xposition = xpos;			yposition = ypos;		}		public function init(path:String,video:String,prent:DisplayObjectContainer) {			rootmc = prent;			vidSrc = path+video;			vidSWF = new Loader();			vidSWF.x = xposition;			vidSWF.y = yposition;			vidSWF.contentLoaderInfo.addEventListener(Event.COMPLETE,LoadCompleteHandler);			vidSWF.load(new URLRequest(path+stVidHolderSwf));		}				private function LoadCompleteHandler(evt:Event):void		{			var loaderInfo:LoaderInfo = evt.target as LoaderInfo;			mcVid = MovieClip(loaderInfo.content);			mcVid.addEventListener("onRevealed",RevealedHandler);			mcVid.addEventListener("onShut",ShutVidHandler);			mcVid.stop();			dispatchEvent(new Event("onLoaded"));			startMedia();		}		private function RevealedHandler(evt:Event):void		{			playbackTimer = new Timer(50,0);			playbackTimer.addEventListener(TimerEvent.TIMER,onTick);			ns.play(vidSrc);			playbackTimer.start();		}		private function ShutVidHandler(evt:Event):void		{			rootmc.removeChild(vidSWF);		}		private function SetUpVideo():void		{			nc = new NetConnection();			nc.addEventListener(NetStatusEvent.NET_STATUS,netStatusHandler);			nc.addEventListener(SecurityErrorEvent.SECURITY_ERROR,securityEventHandler);			nc.connect(null);			ns = new NetStream(nc);			client = new CustomClient();			client.addEventListener("onMetaData",MetaDataHandler);			ns.client = client;			mcVid.vid.attachNetStream(ns); 					}		private function SetUpSwf():void		{			/*			vidSWF = new Loader();			vidSWF.x = xposition;			vidSWF.y = yposition;			vidSWF.contentLoaderInfo.addEventListener(Event.COMPLETE,LoadCompleteHandler);			vidSWF.load(new URLRequest(path+stVidHolderSwf));			*/		}		public function startMedia():void		{			var extension:String = vidSrc.slice(vidSrc.lastIndexOf(".")+1,vidSrc.length);			type = extension;						switch(type)			{				case "flv":					SetUpVideo();				break;				case "swf":					SetUpSwf();					return;				break;			}			rootmc.addChild(vidSWF);			mcVid.vid.visible=true;			mcVid.play();		}		private function onTick(evt:Event):void		{			if(ns.time >= duration){				EndVideo();			}		}		private function EndVideo():void		{			ns.close();			mcVid.vid.visible=false;			mcVid.play();			playbackTimer.stop();		}		private function netStatusHandler(evt:Event):void{		}		private function securityEventHandler(evt:Event):void{		}		private function MetaDataHandler(evt:Event):void		{			duration = client.Duration;		}	}	}import flash.display.Sprite;import flash.events.*;class CustomClient extends Sprite{	public var Duration:Number;		function CustomClient()	{			}	public function onMetaData(info:Object):void{		Duration = info.duration;		dispatchEvent(new Event("onMetaData"));	}		public function onCuePoint(info:Object):void{			}		}