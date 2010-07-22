package net.electroland.kioskengine {
	import flash.events.*;
    import flash.net.URLRequest;
	import flash.display.Loader;
	import flash.media.Sound;
	import flash.media.SoundChannel;
	import flash.media.SoundTransform;
	
	public class AudioItem extends Item{
		private var soundPlayer:Sound;
		private var sound:SoundChannel;
		private var soundVolume:int;
		private var currentFile:String;
		private var pausePoint:int;
		private var skipToPoint:int;
		
		/*
		
		AUDIOITEM.as
		by Aaron Siegel, 7-3-09
		
		Handles only audio for units.
		
		*/
		
		public function AudioItem(conductor:Conductor, itemElement:XML){
			super(conductor);
			parseXML(itemElement);
		}
		
		public function onKeyPressedEvent(event:KeyboardEvent):void{
			//trace(event.keyCode);
			if(event.keyCode == 39){	// right arrow key
				if(sound != null){
					sound.stop();
					sound = soundPlayer.play(skipToPoint);
				}
			}
		}
		
		public function loadContent():void{
			soundPlayer = new Sound();
			soundPlayer.addEventListener(Event.COMPLETE, soundLoaded);
			soundPlayer.addEventListener(IOErrorEvent.IO_ERROR, soundIOError);
			if(itemEntries.hasOwnProperty("all")){
				soundPlayer.load(new URLRequest(String(itemEntries["all"].value)));
				soundVolume = Number(itemEntries["all"].volume) * conductor.getGain();
			} else {
				soundPlayer.load(new URLRequest(String(itemEntries[conductor.getLanguage()].value)));
				soundVolume = Number(itemEntries[conductor.getLanguage()].volume) * conductor.getGain();
			}
		}
		
		public function mute():void{
			if(sound != null){
				var soundTrans:SoundTransform = new SoundTransform(0);
				sound.soundTransform = soundTrans;
			}
		}
		
		public function pause():void{
			if(sound != null){
				pausePoint = sound.position;
				sound.stop();
			}
		}
		
		public function play():void{
			if(sound != null){
				sound = soundPlayer.play(pausePoint);
			}
		}
		
		override public function reset():void{
			if(sound != null){
				sound.stop();
				conductor.logItemEvent(" ", " ", getName(), getDescription(), "sound interrupted");
				trace("SOUND STOPPED");
			}
			conductor.parent.stage.removeEventListener(KeyboardEvent.KEY_DOWN, onKeyPressedEvent);
			super.reset();
		}
		
		private function soundComplete(event:Event):void{
			//trace("audio finished playing");
			conductor.logItemEvent(" ", " ", getName(), getDescription(), "sound complete");
			dispatchEvent(new ItemEvent("audioComplete", "default"));
		}
		
		private function soundIOError(event:Event):void{
			trace("ERROR: failed to load background audio");
		}
		
		private function soundLoaded(event:Event):void{
			startAudioCallback = startAudio;
			stopAudioCallback = pause;
			if(contentLoadedCallback != null){
				contentLoadedCallback(this);	// tell unit this has been loaded
			}
		}
		
		public function startAudio():void{
			trace("SOUND STARTED");
			conductor.logItemEvent(" ", " ", getName(), getDescription(), "sound started");
			skipToPoint = soundPlayer.length - 5000;	// milliseconds
			sound = soundPlayer.play();
			sound.addEventListener(Event.SOUND_COMPLETE, soundComplete);
			var soundTrans:SoundTransform = new SoundTransform(soundVolume/100);
			sound.soundTransform = soundTrans;
			conductor.parent.stage.addEventListener(KeyboardEvent.KEY_DOWN, onKeyPressedEvent);
		}
		
		public function startTimer():void{
			// begin timers on various effects
			startEffectTimers();
		}
		
	}
	
}