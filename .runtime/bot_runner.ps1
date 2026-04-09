$ErrorActionPreference = ''Stop''
$env:PYTHONNOUSERSITE = ''1''
Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
$env:TELEGRAM_BOT_TOKEN = ''8475512450:AAH4mAamAg51kWXjm3_wpI5LvEgYOq_t2G4''
Set-Location ''D:\Projects\gpt_crypto_bot\files''
& ''D:\Projects\gpt_crypto_bot\pyembed\python.exe'' bot.py 1>> ''D:\Projects\gpt_crypto_bot\bot_stdout.log'' 2>> ''D:\Projects\gpt_crypto_bot\bot_stderr.log''
