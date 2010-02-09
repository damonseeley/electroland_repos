package net.electroland.duke {
	
	import flash.display.Sprite;
	
	public class RadioButtonGroup extends Sprite{
		
		public var buttons:Array;
		public var callbackFunction:Function;
		
		public function RadioButtonGroup(x:Number, y:Number, callbackFunction:Function){
			this.x = x;
			this.y = y;
			this.callbackFunction = callbackFunction;
			buttons = new Array();
		}
		
		public function addButton(id:Number, label:String, x:Number, y:Number, activated:Boolean){
			var button = new RadioButton(this, id, label, x, y, activated);
			addChild(button);
			buttons.push(button);
		}
		
		public function buttonSelected(radioButton:RadioButton):void{
			callbackFunction(radioButton.id);
			for(var i:Number = 0; i<buttons.length; i++){
				if(buttons[i] != radioButton){
					buttons[i].deactivate();
				}
			}
		}
		
		public function activate(id:Number):void{
			for(var i:Number = 0; i<buttons.length; i++){
				if(buttons[i].id != id){
					buttons[i].deactivate();
				} else {
					buttons[i].activate();
				}
			}
		}
		
	}
	
}