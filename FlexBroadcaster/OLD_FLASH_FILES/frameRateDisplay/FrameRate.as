
package frameRateDisplay {
	
	import flash.display.*;
	import flash.events.Event;
	import flash.utils.getTimer;
	import flash.utils.Timer;
	import flash.events.TimerEvent;
	import flash.text.TextField;
	

	public class FrameRate {
		private var previousTime:Number;
		private var currentTime:Number;
		private var fps:Number;
		private var stage:Stage;
		private var txtField:TextField;
		private var array:Array =  new Array();

		public function FrameRate(stage:Stage, txtField:TextField)  {
			this.stage = stage;
			this.txtField = txtField;
			currentTime = new Date().valueOf();
			//trace(currentTime);
		}
		
		public function get getFrameRate():Number {
			return fps;
		}

		public function prepareFrameRate() {
			//trace(theStage);
			//theStage.frameRate = 24;
			stage.addEventListener(Event.ENTER_FRAME, onEnterFrame);

		}

		public function onEnterFrame(event:Event):void {
			previousTime = currentTime;
			currentTime = new Date().valueOf();
			
			var msDiff:Number = currentTime - previousTime;
			fps = 1000/msDiff;
			
			var avg:int = 90;
			array.push(fps);
			if (array.length > avg) {
				array = array.slice(1,array.length);
			}
			
			var total:Number;
			total = 0;
			//trace(total);
			for each (var val in array) {
				total+=val;
				//trace(val);
			}

			txtField.text = String((total/array.length));

		}
	}
	
}