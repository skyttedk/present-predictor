
we must use sqllite with the following tables

user (api users must be created here)

  username
  api_key

user_api_call_log ( log all requests to api )

  date_time
  api_route
  request_payload
  response_payload


product_attributes

  product_id
  product_hash
  
  thread_id  ( see open reference-material/ai.txt for what info we must store)
  run_id

  <attributes>...  ( reference-material/classification.schema.json for attributes)


  
