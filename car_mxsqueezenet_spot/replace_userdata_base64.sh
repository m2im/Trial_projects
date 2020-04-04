USER_DATA=`base64 -w 0 user_data_script.sh`
# sed -i '' "s|base64_encoded_bash_script|$USER_DATA|g" $1
sed -i 's|base64_encoded_bash_script|$USER_DATA|g' spot_fleet_config.json
