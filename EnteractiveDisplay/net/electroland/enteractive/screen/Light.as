package net.electroland.enteractive.screen {
	
	import flash.display.MovieClip;
	import flash.geom.ColorTransform;
	
	/*
	LIGHT.as
	by Aaron Siegel, 1-20-2010
	
	This controls the display of each individual simulated light, and reflects
	the current state of the light data sent from the java application.
	
	*/
	
	public class Light extends MovieClip{
		
		public var id:Number;
		public var w:Number;
		public var h:Number;
		public var lightValue:Number;
		
		public function Light(id:Number, x:Number, y:Number, w:Number, h:Number){
			this.id = id;
			this.x = x;
			this.y = y;
			this.w = w;
			this.h = h;
			
			this.graphics.lineStyle(2, 0xff0000, 1);
			//this.graphics.beginFill(0x000000, 0);
			this.graphics.drawRect(0,0,w,h);
			//this.graphics.endFill();
		}
		
		public function updateLight(lightValue:Number):void{
			// receive new lighting value and update appearance
			// TODO: switch this to use alpha instead
			this.lightValue = lightValue;
			var redHex:String = lightValue.toString(16);
			if(redHex.length < 2){
				redHex = "0"+redHex;
			}
			var ct:ColorTransform = this.transform.colorTransform;
			ct.color = uint("0x"+redHex+"0000");
			this.transform.colorTransform = ct;
		}
		
	}
	
}