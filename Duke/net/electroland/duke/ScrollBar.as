package net.electroland.duke {
	
	import flash.display.MovieClip;
	import flash.events.MouseEvent;
	import flash.text.TextField;
	import flash.text.TextFormat;
	
	public class ScrollBar extends MovieClip{
		
		public var label:String;
		public var minVal:Number;
		public var maxVal:Number;
		public var val:Number;
		public var callback:Function;
		public var mouseDown:Boolean = false;
		public var w:Number = 100;
		public var h:Number = 20;
		public var bar:MovieClip;
		
		public var textLabel:TextField;
		public var textFormat:TextFormat;
		public var valueLabel:TextField;
		
		public function ScrollBar(label:String, x:Number, y:Number, minVal:Number, maxVal:Number, val:Number, callback:Function){
			this.label = label;
			this.x = x;
			this.y = y;
			this.minVal = minVal;
			this.maxVal = maxVal;
			this.val = val;
			this.callback = callback;
			
			this.graphics.beginFill(0x999999);
			this.graphics.drawRect(0, 0, w, h);
			this.graphics.endFill();
			
			bar = new MovieClip();
			bar.graphics.beginFill(0x666666);
			var normVal:Number = (val-minVal) / (maxVal-minVal);
			bar.graphics.drawRect(0, 0, normVal * w, h);
			bar.graphics.endFill();
			addChild(bar);
			
			textFormat = new TextFormat("Arial", 12, 0x333333);
			
			textLabel = new TextField();
			textLabel.text = label;
			textLabel.x = w + 4;
			textLabel.y = 0;
			textLabel.width = w;
			textLabel.autoSize = "left";
			textLabel.height = h;
			textLabel.selectable = false;
			textLabel.setTextFormat(textFormat);
			addChild(textLabel);
			
			valueLabel = new TextField();
			valueLabel.text = String(val);
			valueLabel.x = 0;
			valueLabel.y = 0;
			valueLabel.width = w;
			valueLabel.height = h;
			valueLabel.selectable = false;
			valueLabel.setTextFormat(textFormat);
			addChild(valueLabel);
			
			this.addEventListener(MouseEvent.MOUSE_DOWN, mousePressed);
			this.addEventListener(MouseEvent.MOUSE_UP, mouseReleased);
			this.addEventListener(MouseEvent.MOUSE_MOVE, mouseMoved);
		}
		
		public function mousePressed(e:MouseEvent):void{
			if(mouseX >= 0 && mouseX <= w){
				mouseDown = true;
				bar.width = mouseX;
				val = ((maxVal - minVal) * (bar.width / w)) + minVal;
				//valueLabel.text = String(val);
				valueLabel.text = String(Math.round(val*100)/100);
				valueLabel.setTextFormat(textFormat);
				callback(val);
			}
		}
		
		public function mouseReleased(e:MouseEvent):void{
			mouseDown = false;
		}
		
		public function mouseMoved(e:MouseEvent):void{
			if(mouseDown){
				if(mouseX >= 0 && mouseX <= w){
					bar.width = mouseX;
					val = ((maxVal - minVal) * (bar.width / w)) + minVal;
				} else if(mouseX < 0){
					bar.width = 0;
					val = minVal;
				} else if(mouseX > w){
					bar.width = w;
					val = maxVal;
				}
				//valueLabel.text = String(val);
				valueLabel.text = String(Math.round(val*100)/100);
				valueLabel.setTextFormat(textFormat);
				callback(val);
			}
		}
		
		public function getValue():Number{
			return val;
		}
		
		public function setValue(val:Number):void{
			this.val = val;
			var normVal:Number = (val-minVal) / (maxVal-minVal);
			bar.width = normVal * w;
			//valueLabel.text = String(val);
			valueLabel.text = String(Math.round(val*100)/100);
			valueLabel.setTextFormat(textFormat);
		}
		
	}
	
}