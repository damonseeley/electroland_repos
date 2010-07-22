package net.electroland.kioskengine {
	import fl.transitions.Tween;
	import fl.transitions.TweenEvent;
	import fl.transitions.easing.*;
	import flash.events.*;
    import flash.net.URLRequest;
	import flash.display.Loader;
	import flash.media.Sound;
	import flash.media.SoundChannel;
	import flash.media.SoundTransform;
	
	public class BackgroundAudio {
		private var conductor:Conductor;
		private var soundPlayer:Sound;
		private var sound:SoundChannel;
		private var soundVolume:int;
		private var soundTrans:SoundTransform;
		private var currentFile:String;
 		
		/*
		
		BACKGROUNDAUDIO.as
		by Aaron Siegel, 7-1-09
		
		Plays a soundtrack in the background which can be changed by Units.
		Has built in functionality for fading out an old sound and fading in a new one.
		
		*/
		
		public function BackgroundAudio(conductor:Conductor){
			this.conductor = conductor;
		}
		
		public function fadeIn(newSound:SoundChannel):void{
			soundTrans = new SoundTransform(0,0);
			var volumeTween:Tween = new Tween(soundTrans, "volume", Strong.easeOut, 0, soundVolume/100, 3, true);
			volumeTween.addEventListener(TweenEvent.MOTION_CHANGE, volumeTweenListener);
			//newSound.soundTransform = soundTrans;
		}
		
		public function fadeOut(previousSound:SoundChannel):void{
			previousSound.stop();
		}
		
		public function playSound(filename:String, soundVolume:int):void{
			if(filename != "" && (currentFile == null || currentFile != filename)){
				if(sound != null){
					fadeOut(sound);
				}
				soundPlayer = new Sound();
				soundPlayer.addEventListener(Event.COMPLETE, soundLoaded);
				soundPlayer.addEventListener(IOErrorEvent.IO_ERROR, soundIOError);
				soundPlayer.load(new URLRequest(filename));
				this.soundVolume = soundVolume;
				currentFile = filename;
			}
		}
	
		private function soundComplete(event:Event):void{
			//trace("background audio finished playing");
			if(sound != null){
				sound.removeEventListener(Event.SOUND_COMPLETE, soundComplete);
				soundLoaded(event);
			}
		}
		
		private function soundIOError(event:Event):void{
			trace("ERROR: failed to load background audio");
		}
		
		private function soundLoaded(event:Event):void{
			//trace("background audio loaded");
			sound = soundPlayer.play();
			sound.addEventListener(Event.SOUND_COMPLETE, soundComplete);
			var soundTrans:SoundTransform = new SoundTransform(soundVolume/100);
			sound.soundTransform = soundTrans;
			// tween volume from 0 to whatever volume specified
			//fadeIn(sound);
		}
		
		private function volumeTweenListener(event:Event):void{
			sound.soundTransform = soundTrans;
			trace(soundTrans.volume);
		}
		
	}
	
}