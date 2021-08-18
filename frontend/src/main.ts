import Vue from 'vue'
import App from './App.vue'
import './registerServiceWorker'
import router from './router'
import store from './store/index'
import vuetify from './plugins/vuetify'
import "@aws-amplify/ui-vue"
import Amplify from "aws-amplify"
import awsconfig from "./aws-exports"
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import { TiptapVuetifyPlugin } from 'tiptap-vuetify';
import 'tiptap-vuetify/dist/main.css'

Vue.use(TiptapVuetifyPlugin, {
	vuetify, iconsGroup:'mdi'
})

Amplify.configure(awsconfig);

Vue.config.productionTip = false

export const eventBus = new Vue

new Vue({
  router,
  store,
  vuetify,
  render: h => h(App)
}).$mount('#app')
