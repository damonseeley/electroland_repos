package net.electroland.duke {
	
	import flash.display.MovieClip;
	import flash.events.MouseEvent;
	import flash.text.TextField;
	import flash.text.TextFormat;
	import flash.geom.ColorTransform;
	
	public class RadioButton extends MovieClip{
		
		public var group:RadioButtonGroup;
		public var id:Number;
		public var label:String;
		public var textLabel:TextField;
		public var textFormat:TextFormat;
		public var w:Number = 20;
		public var h:Number = 20;
		public var box:MovieClip;
		public var activated:Boolean;
		
		public function RadioButton(group:RadioButtonGroup, id:Number, label:String, x:Number, y:Number, activated:Boolean){
			this.group = group;
			this.id = id;
			this.label = label;
			this.x = x;
			this.y = y;
			this.activated = activated;
			
			box = new MovieClip();
			if(activated){
				box.graphics.beginFill(0x333333);
			} else {
				box.graphics.beginFill(0x999999);
			}
			box.graphics.drawRect(0, 0, w, h);
			box.graphics.endFill();
			addChild(box);
			
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
			
			this.addEventListener(MouseEvent.MOUSE_DOWN, mousePressed);
		}
		
		public function mousePressed(e:MouseEvent):void{
			if(mouseX >= 0 && mouseX <= w){
				// notify group that this was pressed
				group.buttonSelected(this);
				var ct:ColorTransform = box.transform.colorTransform;
				ct.color = 0x333333;
				box.transform.colorTransform = ct;
			}
		}
		
		public function deactivate():void{
			var ct:ColorTransform = box.transform.colorTransform;
			ct.color = 0x999999;
			box.transform.colorTransform = ct;
		}
		
		public function activate():void{
			var ct:ColorTransform = box.transform.colorTransform;
			ct.color = 0x333333;
			box.transform.colorTransform = ct;
		}
		
	}
	
}