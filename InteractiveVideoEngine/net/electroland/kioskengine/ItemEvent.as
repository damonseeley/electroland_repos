package net.electroland.kioskengine {
	import flash.events.Event;
		
	public class ItemEvent extends Event {
		public static const ITEM_EVENT = "itemEvent";
		private var action:String;
		private var args:String;
		private var src:String;
		
		public function ItemEvent(action:String, args:String){
			super(ITEM_EVENT);
			this.action = action;
			this.args = args;
		}
		
		public function getAction():String{
			// button action, ie: move to unit, move to module, button script, etc.
			// video or audio action, on completion.
			return action;
		}
		
		public function getArguments():String{
			// property pertaining to action, such as a link or identifier.
			return args;
		}
		
		public function getSrc():String{
			// property specifically for unit switching, thus not mandatory
			return src;
		}
		
		public function setSrc(src:String):void{
			this.src = src;
		}
		
	}
	
}