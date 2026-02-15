import threading
import time
from datetime import datetime
from typing import List, Optional, Dict
import queue

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData, TickerId

class IBClient(EWrapper, EClient):
    def __init__(self, host: str, port: int, client_id: int):
        EClient.__init__(self, self)
        self.host = host
        self.port = port
        self.client_id = client_id
        
        self.started = False
        self.next_valid_order_id = None
        self._data_queue = queue.Queue()
        self._error_queue = queue.Queue()
        
        # Thread for the message loop
        self.thread = threading.Thread(target=self.run_loop, daemon=True)

    def connect_and_start(self) -> bool:
        try:
            self.connect(self.host, self.port, self.client_id)
            self.thread.start()
            
            # Wait for connection
            start = time.time()
            while not self.isConnected():
                 if time.time() - start > 5:
                     return False
                 time.sleep(0.1)
            
            # wrapper callback 
            # In python API it's usually automatic after connect, but sometimes nextValidId is the confirm
            return True
        except Exception as e:
            print(f"IB Connect Error: {e}")
            return False

    def run_loop(self):
        self.run()

    def error(self, reqId: TickerId, errorCode: int, errorString: str, advancedOrderRejectJson = ""):
        # Filter benign messages
        if errorCode in [2104, 2106, 2158]: 
            # Market data farm connection is OK
            return
        print(f"IB Error {reqId} {errorCode}: {errorString}")
        self._error_queue.put((reqId, errorCode, errorString))

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_valid_order_id = orderId
        self.started = True

    # --- Historical Data Wrapper ---
    def historicalData(self, reqId: int, bar: BarData):
        self._data_queue.put(bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        self._data_queue.put("END")

    # --- Synchronous Methods ---
    def get_historical_data(self, contract: Contract, end_datetime: str, duration: str, bar_size: str, what_to_show: str, use_rth: int) -> List[BarData]:
        if not self.isConnected():
            return []
            
        req_id = 1001 # simplistic req_id
        
        # Clear queue
        with self._data_queue.mutex:
            self._data_queue.queue.clear()
            
        self.reqHistoricalData(req_id, contract, end_datetime, duration, bar_size, what_to_show, use_rth, 1, False, [])
        
        data = []
        try:
            while True:
                item = self._data_queue.get(timeout=10) # 10s timeout
                if item == "END":
                    break
                data.append(item)
        except queue.Empty:
            print(f"Timeout waiting for data for {contract.symbol}")
            self.cancelHistoricalData(req_id)
            
        return data

    def disconnect_and_stop(self):
        if self.isConnected():
            self.disconnect()

# Helper to create contracts
def make_contract(symbol: str, sec_type: str = "STK", exchange: str = "SMART", currency: str = "USD") -> Contract:
    contract = Contract()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.exchange = exchange
    contract.currency = currency
    # Some specific handling might be needed for indices or futures
    if sec_type == "IND":
        contract.exchange = "CBOE" if symbol == "VIX" else exchange
    return contract
