### Node-red components in the project
- node-red (core, ex. inject, function, mqtt etc.)
- node-red-dashboard v3.6.5
- node-red-node-ui-table v0.4.4



### JS script for function blocks in the project
Generate payload from MQTT message
```
var tableData = flow.get("savedData") || [];
tableData.unshift(msg.payload);
msg.payload = tableData;
flow.set("savedData", tableData);
return msg;
```

Update an UI Table
```
msg.payload = 
{
    "command": "updateRow" // or "updateRow", "replaceData", etc.
    // "arguments": [ ... ] // Optional arguments for the command
}
return msg;
```

Clear table
```
flow.set("savedData", null);
msg.payload = []
return msg;
```


