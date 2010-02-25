package net.electroland.enteractive.screen {
	
	import flash.display.Sprite;
	import flash.events.*;
	import flash.net.XMLSocket;
	
	public class XMLServer extends Sprite {
		
		private var hostName:String = "localhost";
        private var port:uint = 10002;
        private var socket:XMLSocket;
		private var lightMatrix:LightMatrix;
		
		public function XMLServer(lightMatrix:LightMatrix){
			this.lightMatrix = lightMatrix;
			socket = new XMLSocket();
			socket.addEventListener(Event.CLOSE, closeHandler);
            socket.addEventListener(Event.CONNECT, connectHandler);
            socket.addEventListener(DataEvent.DATA, dataHandler);
            socket.addEventListener(IOErrorEvent.IO_ERROR, ioErrorHandler);
            socket.addEventListener(ProgressEvent.PROGRESS, progressHandler);
            socket.addEventListener(SecurityErrorEvent.SECURITY_ERROR, securityErrorHandler);
			socket.connect(hostName, port);
		}
		
		private function closeHandler(event:Event):void {
            trace("closeHandler: " + event);
        }

        private function connectHandler(event:Event):void {
            trace("connectHandler: " + event);
        }

        private function dataHandler(event:DataEvent):void {
            //trace("dataHandler: " + event);
			var xmlData:XML = new XML(event.data);
			lightMatrix.updateLights(xmlData);
        }

        private function ioErrorHandler(event:IOErrorEvent):void {
            trace("ioErrorHandler: " + event);
        }

        private function progressHandler(event:ProgressEvent):void {
            trace("progressHandler loaded:" + event.bytesLoaded + " total: " + event.bytesTotal);
        }

        private function securityErrorHandler(event:SecurityErrorEvent):void {
            trace("securityErrorHandler: " + event);
        }

		
	}
	
}