<template>
	<v-btn id="setting-btn"
		class="ma-2"
		rounded
		outlined
		color="grey"
		@click="settingPage">
		계정설정
	</v-btn>
</template>

<script lang="ts">
	import Vue from 'vue'

	export default Vue.extend({
		methods:{
			settingPage(){
				location.href="/setting"
			}
		}
	})
</script>

<style>
#setting-btn:hover{
	color:#757575 !important;
}
</style>