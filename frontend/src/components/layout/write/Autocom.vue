<template>
	<div style="height:45%;">
	  <div v-if="toggleSignAuto" style="margin-left:8px;position:relative;">
		<v-tooltip left>
		  <template v-slot:activator="{ on, attrs }">
			<v-btn style="position:absolute;right:4px;" @click="showAutocom" icon v-bind="attrs" v-on="on">
			  <v-icon>mdi-chevron-left</v-icon>
			</v-btn>
		  </template>
		  <span>자동완성</span>
		</v-tooltip>
	  </div>
	  <div v-else style="margin-left:8px;position:relative;">
		<span><v-tooltip top>
		  <template v-slot:activator="{ on, attrs }">
			<v-btn icon v-bind="attrs" v-on="on" @click="autoComplete"><v-icon>mdi-motion-play-outline</v-icon></v-btn>
		  </template>
		  <span>문장 자동완성</span>
		</v-tooltip></span>
		<span style="position:absolute;right:4px;"><v-btn @click="showAutocom" icon><v-icon>mdi-chevron-down</v-icon></v-btn></span>
		<div style="margin-left:8px;margin-right:14px;margin-top:10px;">
		  <v-textarea class="mx-auto"
		  	:value="text"
			solo
			no-resize
			placeholder="AI 자동완성이 이곳에 출력됩니다.."
			height="32vh"
			readonly>
		  </v-textarea>
		</div>
	  </div>
	</div>
</template>

<script lang="ts">
	import { Component, Vue } from 'vue-property-decorator';
	import { namespace } from 'vuex-class';

	const WriteStoreModule = namespace('writeStore')

	@Component
	export default class Autocom extends Vue {
		toggleSignAuto = true
		text = ''

		@WriteStoreModule.Getter('getItemList')
		private itemList!:any[]

		@WriteStoreModule.Getter('getActiveVal')
		private activeVal!:number

		showAutocom():void{
			this.toggleSignAuto = !this.toggleSignAuto
		}

		autoComplete():void{
			var sentenceList = []

			var content = ''
			var textList = []
			var text = ''

			if(this.activeVal === 1){
				content = this.itemList[0].text.replaceAll("\n"," ")
			}else if(this.activeVal > 1){
				for(var i=0; i<this.itemList[0].children.length;i++){
					if(this.activeVal === this.itemList[0].children[i].id){
						content = this.itemList[0].children[i].text.replaceAll("\n", " ")
					}
				}
			}
			textList = content.split(" ").slice(Math.max(content.split(" ").length - 8, 0))
			text = textList.join(" ")
			console.log(text)



			//this.text = sentenceList
		}
	}
</script>

<style scoped>

</style>